"""Generate region-proposal candidates with a cloud vision-language API.

The project owner authorized uploading these frame images to the API provider
on 2026-09-20; every other pipeline stage stays local and offline. The cloud
model only returns text, which is parsed strictly as a JSON candidate list:
nothing in the response is executed, merged into the navigation stack, or
treated as a safety decision, and raw responses are archived next to the
proposal document for audit. Unlike the local greedy decoder, a cloud API is
not bit-reproducible: the provider may update the model or sampling at any
time, so provenance rests on the archived raw responses rather than on
determinism. The API key is read from an environment variable and never
enters command lines, source files, or output documents.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from field_boundary_mapping.evidence.review_results import ResultStore
from field_boundary_mapping.proposals.run_offline import write_json
from field_boundary_mapping.proposals.run_proposals import identifier
from field_boundary_mapping.perception.vlm_proposer import INSTRUCTION, PROMPT_VERSION as INSTRUCTION_VERSION, parse_proposals

BASE = Path(__file__).resolve().parents[2]   # 仓库根
PROMPT_VERSION = "cloud-vlm-proposals-v2"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_TIMEOUT = 120.0


class CloudVLM:
    def __init__(self, base_url, model, api_key_env, max_tokens, timeout):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key_env = api_key_env
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"API key environment variable {api_key_env} is unset or empty")
        self._api_key = api_key
        self.metadata = {
            "backend": "cloud_api_openai_compatible", "base_url": self.base_url,
            "model": model, "api_key_env": api_key_env,
            "prompt_version": PROMPT_VERSION,
            "instruction_version": f"{INSTRUCTION_VERSION} (identical text as the local proposer)",
            "vlm_used": True, "max_tokens": max_tokens, "temperature": 0,
            "data_egress": "frame images are uploaded to the provider; authorized by the project owner 2026-09-20",
            "reproducibility": "cloud APIs are not bit-reproducible; archived raw responses are the provenance",
            "score_meaning": "none; the model emits text boxes only, no calibration and no safety probability",
        }

    def ask(self, image_bytes):
        return self.ask_images([(image_bytes, "image/jpeg")], INSTRUCTION)

    def ask_images(self, images, instruction, response_key="proposals", *, response_format=None, strict_content=False):
        if not isinstance(images, list) or not 1 <= len(images) <= 4:
            raise ValueError("images must be a list of 1 to 4 (image_bytes, mime_type) tuples")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        if not isinstance(response_key, str) or not response_key.strip():
            raise ValueError("response_key must be a nonempty string")
        if type(strict_content) is not bool:
            raise ValueError("strict_content must be boolean")
        if response_format is not None:
            if not isinstance(response_format, dict) or response_format != {"type": "json_object"}:
                raise ValueError("response_format must be None or {'type': 'json_object'}")
            if not re.search(r"\bjson\b", instruction, re.IGNORECASE):
                raise ValueError("JSON mode requires the word JSON in instruction")
        for image in images:
            if not isinstance(image, tuple) or len(image) != 2:
                raise ValueError("Each image must be an (image_bytes, mime_type) tuple")
            image_bytes, mime_type = image
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ValueError("image_bytes must be nonempty bytes")
            if mime_type not in ("image/jpeg", "image/png"):
                raise ValueError("mime_type must be image/jpeg or image/png")

        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime_type};base64,"
                                  + base64.b64encode(image_bytes).decode("ascii")}}
            for image_bytes, mime_type in images
        ]
        content.append({"type": "text", "text": instruction})
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self.model.lower().startswith("qwen3"):
            payload["enable_thinking"] = False
        if response_format is not None:
            payload["response_format"] = response_format.copy()
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST")
        started = time.perf_counter()
        attempts = []
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:2000]
                raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                # Response-phase disconnects (RemoteDisconnected) surface raw from
                # http.client, while connect-phase ones arrive wrapped in URLError.
                attempts.append(f"{type(exc).__name__}: {exc}")
                if attempt == 2:
                    raise RuntimeError(f"API request failed after 3 attempts: {attempts}") from exc
                time.sleep(2 ** attempt)
        seconds = time.perf_counter() - started
        message = body["choices"][0]["message"]
        content = message.get("content")
        response_text = content if strict_content or isinstance(content, str) \
            else ("" if content is None else json.dumps(content, ensure_ascii=False))
        reasoning = message.get("reasoning_content")
        parsed_from = "content"
        if not strict_content and response_key not in response_text and isinstance(reasoning, str) and response_key in reasoning:
            response_text, parsed_from = reasoning, "reasoning_content"
        usage = body.get("usage") or {}
        timing = {
            "seconds": seconds, "model": body.get("model"), "request_id": body.get("id"),
            "parsed_from": parsed_from, "finish_reason": body["choices"][0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        }
        if strict_content:
            timing["strict_content"] = True
        if isinstance(reasoning, str) and reasoning:
            timing["reasoning_content"] = reasoning
        return response_text, timing


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Proposal output must be a new file; existing files are never overwritten")
    store = ResultStore(args.results)
    manifest_path = store.resolve("manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    frames = {}
    for frame in manifest["frames"]:
        identifier(frame.get("frame_id"), "source frame_id")
        if frame["frame_id"] in frames:
            raise ValueError("Duplicate source frame_id")
        frames[frame["frame_id"]] = frame
    selected = []
    for frame_id in args.frames:
        identifier(frame_id, "frame_id")
        if frame_id not in frames:
            raise ValueError(f"Unknown source frame_id: {frame_id}")
        selected.append(frames[frame_id])
    if not 1 <= len(selected) <= 16:
        raise ValueError("Select 1 to 16 frames explicitly")

    model = CloudVLM(args.base_url, args.model, args.api_key_env, args.max_output_tokens, args.timeout)
    document = {"schema_version": "field_boundary_proposals_v1", "proposal_source": "cloud_vlm", "frames": []}
    audit = {"prompt_version": PROMPT_VERSION, "model": model.metadata, "instruction": INSTRUCTION, "frames": []}
    for frame in selected:
        frame_id = frame["frame_id"]
        image_bytes, _ = store.read_file(frame["image_path"])
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != frame.get("artifact_sha256", {}).get(frame["image_path"]):
            raise ValueError(f"Source manifest hash mismatch for {frame_id}")
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Cannot decode image for {frame_id}")
        height, width = decoded.shape[:2]
        print(f"cloud inference {frame_id} ({width}x{height}) via {args.model}", flush=True)
        response, timing = model.ask(image_bytes)
        proposals, rejected = parse_proposals(response, width, height)
        print(f"  {len(proposals)} accepted, {len(rejected)} rejected", flush=True)
        document["frames"].append({
            "frame_id": frame_id, "image_sha256": digest,
            "proposals": [{key: value for key, value in item.items() if key != "model_box_raw"} for item in proposals],
        })
        audit["frames"].append({
            "frame_id": frame_id, "image_sha256": digest, "image_width": width, "image_height": height,
            "raw_response": response, "timing": timing,
            "accepted": proposals, "rejected": rejected,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, document)
    write_json(output.with_name(output.stem + "_raw_vlm_outputs.json"), audit)
    print(f"saved {output} and raw outputs; {sum(len(f['proposals']) for f in document['frames'])} proposals", flush=True)
    return document


def main():
    parser = argparse.ArgumentParser(description="Cloud VLM region proposals via an OpenAI-compatible API; JSON boxes only, no tool execution")
    parser.add_argument("--results", type=Path, required=True, help="Existing source run with hashed original images")
    parser.add_argument("--frames", nargs="+", required=True, help="1 to 16 explicit source frame_ids")
    parser.add_argument("--model", required=True, help="Provider model name, e.g. qwen-vl-max-latest")
    parser.add_argument("--output", type=Path, required=True, help="New proposals JSON path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
                        help="Environment variable holding the API key; the key never enters arguments or files")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
