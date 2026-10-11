import argparse
import base64
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_boundary_mapping.perception.cloud_vlm_proposer import CloudVLM, DEFAULT_API_KEY_ENV, run
from field_boundary_mapping.perception.vlm_proposer import INSTRUCTION


PROPOSALS_JSON = ('{"proposals": [{"semantic": "crop", "box_xyxy": [10, 20, 500, 900],'
                  ' "relation": "unspecified", "rationale": "center field rows"}]}')


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def api_body(content=PROPOSALS_JSON, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"id": "chatcmpl-test", "model": "provider-model",
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 56}}


def urllib_error_count(message):
    return message.count("RemoteDisconnected")


class CloudVLMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="field-cloud-vlm-test-")
        self.addCleanup(self.temp.cleanup)
        self.env = {DEFAULT_API_KEY_ENV: "test-key"}
        self.image_bytes = b"fake-jpeg-bytes"

    def make_model(self, model="qwen-vl-max-latest", **changes):
        options = {"base_url": "https://provider.example/v1", "model": model,
                   "api_key_env": DEFAULT_API_KEY_ENV, "max_tokens": 1024, "timeout": 30.0}
        options.update(changes)
        with patch.dict(os.environ, self.env, clear=False):
            return CloudVLM(**options)

    def test_missing_api_key_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                CloudVLM("https://provider.example/v1", "qwen-vl-max-latest",
                         DEFAULT_API_KEY_ENV, 1024, 30.0)

    def test_ask_builds_payload_and_parses_response(self):
        for model, expect_thinking in (("qwen-vl-max-latest", False), ("qwen3-vl-plus", True)):
            with self.subTest(model=model):
                cloud = self.make_model(model=model)
                with patch("urllib.request.urlopen", return_value=FakeResponse(api_body())) as urlopen:
                    response_text, timing = cloud.ask(self.image_bytes)
                request = urlopen.call_args[0][0]
                self.assertTrue(request.full_url.endswith("/chat/completions"))
                self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                payload = json.loads(request.data)
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["max_tokens"], 1024)
                self.assertEqual(("enable_thinking" in payload), expect_thinking)
                if expect_thinking:
                    self.assertFalse(payload["enable_thinking"])
                image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
                self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
                text_part = payload["messages"][0]["content"][1]
                self.assertEqual(text_part["type"], "text")
                self.assertEqual(text_part["text"], INSTRUCTION)
                self.assertEqual(response_text, PROPOSALS_JSON)
                self.assertEqual(timing["parsed_from"], "content")
                self.assertEqual(timing["request_id"], "chatcmpl-test")
                self.assertEqual(timing["prompt_tokens"], 1234)
                self.assertEqual(timing["finish_reason"], "stop")
                self.assertNotIn("reasoning_content", timing)
                legacy_payload = {
                    "model": model, "temperature": 0, "max_tokens": 1024,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64,"
                                   + base64.b64encode(self.image_bytes).decode("ascii")}},
                        {"type": "text", "text": INSTRUCTION},
                    ]}],
                }
                if expect_thinking:
                    legacy_payload["enable_thinking"] = False
                self.assertEqual(request.data, json.dumps(legacy_payload).encode("utf-8"))

    def test_strict_content_never_substitutes_reasoning_or_stringifies(self):
        cloud = self.make_model()
        for content in ("invalid final text", None, {"regions": []}, [{"regions": []}]):
            body = api_body(content=content, reasoning=PROPOSALS_JSON)
            with self.subTest(content=content), patch("urllib.request.urlopen", return_value=FakeResponse(body)):
                raw, timing = cloud.ask_images([(self.image_bytes, "image/jpeg")], "Return JSON.", strict_content=True)
            self.assertEqual(raw, content)
            self.assertEqual(timing["parsed_from"], "content")
            self.assertIs(timing["strict_content"], True)
            self.assertEqual(timing["reasoning_content"], PROPOSALS_JSON)
        with patch("urllib.request.urlopen") as send:
            with self.assertRaises(ValueError):
                cloud.ask_images([(self.image_bytes, "image/jpeg")], "Return JSON.", strict_content=1)
            send.assert_not_called()

    def test_ask_delegates_to_ask_images(self):
        cloud = self.make_model()
        expected = (PROPOSALS_JSON, {"finish_reason": "stop"})
        with patch.object(cloud, "ask_images", return_value=expected) as ask_images:
            self.assertIs(cloud.ask(self.image_bytes), expected)
        ask_images.assert_called_once_with([(self.image_bytes, "image/jpeg")], INSTRUCTION)

    def test_ask_images_preserves_order_mime_bytes_and_custom_instruction(self):
        cloud = self.make_model()
        images = [(b"\x89PNG\r\n\x1a\nfirst\x00", "image/png"),
                  (b"\xff\xd8second\xff\xd9", "image/jpeg")]
        instruction = "  Identify agricultural regions in chronological image order.\n"
        regions = '{"regions": []}'
        with patch("urllib.request.urlopen", return_value=FakeResponse(api_body(regions))) as urlopen:
            response_text, timing = cloud.ask_images(images, instruction, response_key="regions")
        urlopen.assert_called_once()
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 30.0})
        payload = json.loads(urlopen.call_args[0][0].data)
        self.assertEqual(payload, {
            "model": cloud.model, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,"
                           + base64.b64encode(image_bytes).decode("ascii")}}
                for image_bytes, mime_type in images
            ] + [{"type": "text", "text": instruction}]}],
        })
        self.assertEqual(response_text, regions)
        self.assertEqual(timing["parsed_from"], "content")
        self.assertEqual(timing["finish_reason"], "stop")

    def test_json_mode_payload_preserves_caller_format_and_token_budget(self):
        for model in ("qwen-vl-max-latest", "qwen3-vl-plus"):
            for instruction in ("JSON", 'Return "json".', "Return JsOn."):
                with self.subTest(model=model, instruction=instruction):
                    cloud = self.make_model(model=model, max_tokens=321)
                    response_format = {"type": "json_object"}
                    with patch("urllib.request.urlopen", return_value=FakeResponse(api_body())) as urlopen:
                        response_text, timing = cloud.ask_images(
                            [(self.image_bytes, "image/jpeg")], instruction,
                            response_format=response_format)
                    urlopen.assert_called_once()
                    payload = json.loads(urlopen.call_args[0][0].data)
                    expected = {
                        "model": model, "temperature": 0, "max_tokens": 321,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": "data:image/jpeg;base64,"
                                       + base64.b64encode(self.image_bytes).decode("ascii")}},
                            {"type": "text", "text": instruction},
                        ]}],
                        "response_format": {"type": "json_object"},
                    }
                    if model.startswith("qwen3"):
                        expected["enable_thinking"] = False
                    self.assertEqual(payload, expected)
                    self.assertEqual(response_format, {"type": "json_object"})
                    self.assertEqual(response_text, PROPOSALS_JSON)
                    self.assertEqual(timing["finish_reason"], "stop")

    def test_explicit_none_keeps_legacy_payload_without_json_instruction(self):
        cloud = self.make_model()
        images = [(self.image_bytes, "image/jpeg")]
        with patch("urllib.request.urlopen", return_value=FakeResponse(api_body())) as urlopen:
            cloud.ask_images(images, "Describe the scene.")
            cloud.ask_images(images, "Describe the scene.", response_format=None)
        self.assertEqual(urlopen.call_count, 2)
        default_data, explicit_data = [call.args[0].data for call in urlopen.call_args_list]
        self.assertEqual(explicit_data, default_data)
        self.assertNotIn("response_format", json.loads(explicit_data))

    def test_response_format_is_keyword_only(self):
        cloud = self.make_model()
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(TypeError):
                cloud.ask_images([(self.image_bytes, "image/jpeg")], "Return JSON.",
                                 "proposals", {"type": "json_object"})
            urlopen.assert_not_called()

    def test_invalid_response_formats_rejected_without_network(self):
        cloud = self.make_model()
        invalid_formats = [
            {}, {"type": "json_schema"},
            {"type": "json_schema", "json_schema": {"name": "scene", "schema": {}}},
            {"type": "json_object", "strict": True},
            {"type": "json_object", "json_schema": {}},
            {"type": "JSON_OBJECT"}, {"type": "text"}, {"type": None},
            {"type": ["json_object"]}, {"other": "json_object"},
            "json_object", '{"type":"json_object"}', [], [("type", "json_object")],
            ("json_object",), True, False, 0, 1, 1.5,
        ]
        for response_format in invalid_formats:
            with self.subTest(response_format=response_format):
                with patch("urllib.request.urlopen") as urlopen:
                    with self.assertRaisesRegex(ValueError, "response_format"):
                        cloud.ask_images([(self.image_bytes, "image/jpeg")], "Return JSON.",
                                         response_format=response_format)
                    urlopen.assert_not_called()

    def test_json_mode_requires_json_word_without_network(self):
        cloud = self.make_model()
        for instruction in ("Describe the scene.", "Return JSONish.",
                            "Use json_object.", "Return nonJSON."):
            with self.subTest(instruction=instruction):
                with patch("urllib.request.urlopen") as urlopen:
                    with self.assertRaisesRegex(ValueError, "word JSON"):
                        cloud.ask_images([(self.image_bytes, "image/jpeg")], instruction,
                                         response_format={"type": "json_object"})
                    urlopen.assert_not_called()

    def test_json_mode_keeps_truncated_response_and_finish_reason_for_caller(self):
        cloud = self.make_model()
        truncated = '{"regions": ['
        body = api_body(content=truncated)
        body["choices"][0]["finish_reason"] = "length"
        with patch("urllib.request.urlopen", return_value=FakeResponse(body)) as urlopen:
            response_text, timing = cloud.ask_images(
                [(self.image_bytes, "image/jpeg")], "Return JSON regions.",
                response_key="regions", response_format={"type": "json_object"})
        urlopen.assert_called_once()
        self.assertEqual(response_text, truncated)
        self.assertEqual(timing["finish_reason"], "length")
        self.assertEqual(timing["parsed_from"], "content")

    def test_json_mode_http_rejection_has_no_retry_or_legacy_fallback(self):
        cloud = self.make_model()
        response_format = {"type": "json_object"}
        error = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions", 400, "Bad Request",
            {}, io.BytesIO(b'{"error":{"message":"response_format unsupported"}}'))
        with patch("urllib.request.urlopen", side_effect=[error, FakeResponse(api_body())]) as urlopen, \
                patch("time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "API HTTP 400:.*response_format unsupported"):
                cloud.ask_images([(self.image_bytes, "image/jpeg")], "Return JSON.",
                                 response_format=response_format)
            urlopen.assert_called_once()
            sleep.assert_not_called()
        self.assertEqual(json.loads(urlopen.call_args[0][0].data)["response_format"],
                         {"type": "json_object"})
        self.assertEqual(response_format, {"type": "json_object"})

    def test_ask_images_accepts_four_images(self):
        cloud = self.make_model()
        images = [(bytes([index]), "image/jpeg") for index in range(4)]
        with patch("urllib.request.urlopen", return_value=FakeResponse(api_body())) as urlopen:
            cloud.ask_images(images, INSTRUCTION)
        content = json.loads(urlopen.call_args[0][0].data)["messages"][0]["content"]
        self.assertEqual(len(content), 5)
        for part, (image_bytes, mime_type) in zip(content, images):
            self.assertEqual(part["image_url"]["url"],
                             f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("ascii"))
        self.assertEqual(content[-1], {"type": "text", "text": INSTRUCTION})

    def test_ask_images_uses_custom_response_key_for_reasoning_fallback(self):
        cloud = self.make_model()
        regions = '{"regions": []}'
        cases = [
            ("", regions, regions, "reasoning_content"),
            (None, regions, regions, "reasoning_content"),
            (PROPOSALS_JSON, regions, regions, "reasoning_content"),
            (regions, PROPOSALS_JSON, regions, "content"),
            ("", PROPOSALS_JSON, "", "content"),
        ]
        for content, reasoning, expected, parsed_from in cases:
            with self.subTest(content=content, reasoning=reasoning):
                body = api_body(content=content, reasoning=reasoning)
                with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
                    response_text, timing = cloud.ask_images(
                        [(self.image_bytes, "image/jpeg")], "Identify regions.", response_key="regions")
                self.assertEqual(response_text, expected)
                self.assertEqual(timing["parsed_from"], parsed_from)
                self.assertEqual(timing["reasoning_content"], reasoning)

    def test_ask_images_retains_non_stop_finish_reason(self):
        cloud = self.make_model()
        for finish_reason in ("length", "content_filter", None):
            with self.subTest(finish_reason=finish_reason):
                body = api_body(content="", reasoning='{"regions": []}')
                body["choices"][0]["finish_reason"] = finish_reason
                with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
                    response_text, timing = cloud.ask_images(
                        [(self.image_bytes, "image/png")], "Identify regions.", response_key="regions")
                self.assertEqual(response_text, '{"regions": []}')
                self.assertEqual(timing["finish_reason"], finish_reason)

    def test_ask_images_rejects_invalid_inputs_without_network(self):
        cloud = self.make_model()
        image = (self.image_bytes, "image/jpeg")
        invalid_images = [
            None, self.image_bytes, (image,), [], [image] * 5,
            [None], [self.image_bytes], [()], [(self.image_bytes,)],
            [(self.image_bytes, "image/jpeg", "extra")], [list(image)],
            [(b"", "image/jpeg")], [(None, "image/jpeg")], [("text", "image/jpeg")],
            [(bytearray(b"data"), "image/jpeg")],
            [(self.image_bytes, "image/gif")], [(self.image_bytes, "image/webp")],
            [(self.image_bytes, "IMAGE/JPEG")], [(self.image_bytes, "")],
            [(self.image_bytes, None)], [image, (b"", "image/png")],
        ]
        cases = [{"images": images} for images in invalid_images]
        cases += [{"instruction": value} for value in (None, "", " \n\t", 123, b"text")]
        cases += [{"response_key": value} for value in (None, "", " \n\t", 123, b"regions")]
        for changes in cases:
            with self.subTest(changes=changes):
                options = {"images": [image], "instruction": "Identify regions.", "response_key": "regions"}
                options.update(changes)
                with patch("urllib.request.urlopen") as urlopen:
                    with self.assertRaises(ValueError):
                        cloud.ask_images(**options)
                    urlopen.assert_not_called()

    def test_ask_falls_back_to_reasoning_content_when_answer_is_empty(self):
        cloud = self.make_model()
        body = api_body(content="", reasoning=PROPOSALS_JSON)
        with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
            response_text, timing = cloud.ask(self.image_bytes)
        self.assertEqual(response_text, PROPOSALS_JSON)
        self.assertEqual(timing["parsed_from"], "reasoning_content")
        self.assertEqual(timing["reasoning_content"], PROPOSALS_JSON)

    def test_http_error_surfaces_provider_detail(self):
        cloud = self.make_model()
        error = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions", 404, "Not Found",
            {}, io.BytesIO(b'{"error":{"message":"model not found"}}'))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as ctx:
                cloud.ask(self.image_bytes)
        self.assertIn("404", str(ctx.exception))
        self.assertIn("model not found", str(ctx.exception))

    def test_transient_disconnects_are_retried_then_succeed(self):
        import http.client
        cloud = self.make_model()
        disconnect = http.client.RemoteDisconnected("Remote end closed connection without response")
        effects = [disconnect, disconnect, FakeResponse(api_body())]
        with patch("urllib.request.urlopen", side_effect=effects) as urlopen, \
                patch("time.sleep") as sleep:
            response_text, _timing = cloud.ask(self.image_bytes)
        self.assertEqual(response_text, PROPOSALS_JSON)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_persistent_disconnect_fails_with_attempt_log(self):
        import http.client
        cloud = self.make_model()
        disconnect = http.client.RemoteDisconnected("Remote end closed connection without response")
        with patch("urllib.request.urlopen", side_effect=disconnect), \
                patch("time.sleep") as sleep:
            with self.assertRaises(RuntimeError) as ctx:
                cloud.ask(self.image_bytes)
        self.assertIn("after 3 attempts", str(ctx.exception))
        self.assertIn("RemoteDisconnected", str(ctx.exception))
        self.assertEqual(urllib_error_count(str(ctx.exception)), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_http_error_is_not_retried(self):
        cloud = self.make_model()
        error = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions", 500, "Server Error",
            {}, io.BytesIO(b"{}"))
        with patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaises(RuntimeError):
                cloud.ask(self.image_bytes)
        self.assertEqual(urlopen.call_count, 1)

    def test_run_refuses_existing_output(self):
        existing = Path(self.temp.name) / "proposals.json"
        existing.write_bytes(b"sentinel")
        args = argparse.Namespace(output=existing)
        with self.assertRaises(ValueError):
            run(args)
        self.assertEqual(existing.read_bytes(), b"sentinel")


if __name__ == "__main__":
    unittest.main()
