import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_vlm_proposer import DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL
from review_results import ResultStore
from run_offline import file_hash, write_json
from run_scene_understanding import (
    check_identity, check_record, instruction_for, load_scene_run, main, run, select_context, source_frames,
)
from scene_understanding import (GATES, INSTRUCTIONS, PROMPT_VERSION, parse_scene, pixel_points,
                                 region_geometry_kind, region_outline, scene_queries)


def sample_scene():
    return {"regions": [
        {"region_id": "r1", "role": "current_field", "surface": "crop",
         "polygon": [[0, 300], [500, 300], [500, 1000], [0, 1000]],
         "evidence": "Visible rows in the vehicle working direction", "uncertainties": ["Ownership unknown"]},
        {"region_id": "r2", "role": "headland_candidate", "surface": "ground",
         "polygon": [[0, 200], [500, 200], [500, 300], [0, 300]],
         "evidence": "Visible strip at row ends", "uncertainties": ["Turning clearance unknown"]}],
        "relations": [{"subject": "r2", "relation": "headland_of", "object": "r1", "evidence": "Strip beyond row ends"}],
        "boundaries": [{"boundary_id": "b1", "side_a": "r1", "side_b": "r2", "kind": "field_headland",
                        "points": [[0, 300], [500, 300]], "evidence": "Visible row-end line"}],
        "unknowns": ["Metric clearance is unknown"]}


def sample_box_scene():
    scene = sample_scene()
    for region, box in zip(scene["regions"], ([0, 300, 500, 1000], [0, 200, 500, 300])):
        del region["polygon"]
        region["box"] = box
    return scene


class SceneSchemaTests(unittest.TestCase):
    def test_valid_scene_round_trip(self):
        scene = sample_scene()
        self.assertEqual(parse_scene(json.dumps(scene)), scene)

    def test_empty_scene_requires_reason(self):
        scene = {"regions": [], "relations": [], "boundaries": [], "unknowns": ["Occluded"]}
        self.assertEqual(parse_scene(json.dumps(scene)), scene)
        scene["unknowns"] = []
        with self.assertRaises(ValueError):
            parse_scene(json.dumps(scene))

    def test_role_and_surface_queries_stay_separate(self):
        scene = sample_scene()
        queries = scene_queries(scene, 1080, 1280)
        self.assertEqual(queries[1]["semantic"], "ground")
        self.assertEqual(queries[1]["proposal_id"], "r2")
        self.assertEqual(queries[1]["box_xyxy"], [0, 256, 540, 384])
        self.assertNotIn("headland", queries[1]["prompt"])
        self.assertTrue(all(query["relation"] == "unspecified" for query in queries))

    def test_mixed_unknown_surfaces_remain_unsegmented(self):
        scene = sample_scene()
        scene["regions"][0]["surface"] = "mixed"
        scene["regions"][1]["surface"] = "unknown"
        self.assertEqual(scene_queries(scene, 100, 100), [])

    def test_pixel_extent_including_image_edge(self):
        self.assertEqual(pixel_points([[0, 0], [1000, 1000]], 1080, 1280), [[0, 0], [1079, 1279]])
        queries = scene_queries(sample_scene(), 1080, 1280)
        self.assertEqual(queries[0]["box_xyxy"][3], 1280)

    def test_rejects_extra_control_fields(self):
        for field in ("semantic_verified", "maneuver_verified", "command", "world_map"):
            scene = sample_scene()
            scene[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_rejects_invalid_roles_and_surfaces(self):
        for field, value in (("role", "safe_turn"), ("surface", "headland_candidate"), ("role", []),
                             ("surface", 5), ("region_id", "../evil"), ("region_id", "unknown")):
            scene = sample_scene()
            scene["regions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_rejects_invalid_polygons(self):
        polygons = [[], [[0, 314, 1000, 620]], [15, 338, 55, 368, 45, 362, 10, 345],
                    [[0, 0], [100, 100]], [[0, 0], [50, 50], [100, 100]],
                    [[0, 0], [1001, 0], [0, 100]], [[False, 0], [100, 0], [0, 100]],
                    [[0.0, 0], [100, 0], [0, 100]], [[0, 0], [100, 0], [0, 100], [0, 0]],
                    [[0, 0], [100, 100], [100, 0], [0, 100]],
                    [[0, 0], [100, 0], [0, 80], [100, 100], [50, 10]],
                    [[float("nan"), 0], [100, 0], [0, 100]]]
        for polygon in polygons:
            scene = sample_scene()
            scene["regions"][0]["polygon"] = polygon
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_box_regions_round_trip_without_inventing_a_polygon(self):
        scene = sample_box_scene()
        parsed = parse_scene(json.dumps(scene))
        self.assertEqual(parsed, scene)
        self.assertNotIn("polygon", json.dumps(parsed))

    def test_box_queries_use_the_same_envelope_math_as_polygons(self):
        box_scene, polygon_scene = sample_box_scene(), sample_scene()
        for width, height in ((1080, 1280), (1000, 1000), (1023, 771)):
            self.assertEqual(
                [query["box_xyxy"] for query in scene_queries(box_scene, width, height)],
                [query["box_xyxy"] for query in scene_queries(polygon_scene, width, height)])

    def test_region_outline_expands_a_box_to_a_rectangle(self):
        region = sample_box_scene()["regions"][0]
        self.assertEqual(region_geometry_kind(region), "vlm_box")
        self.assertEqual(region_outline(region, 1080, 1280),
                         pixel_points([[0, 300], [500, 300], [500, 1000], [0, 1000]], 1080, 1280))
        polygon_region = sample_scene()["regions"][0]
        self.assertEqual(region_geometry_kind(polygon_region), "vlm_polygon")
        self.assertEqual(region_outline(polygon_region, 1080, 1280),
                         pixel_points(polygon_region["polygon"], 1080, 1280))

    def test_rejects_invalid_boxes(self):
        for box in ([], [0, 300, 500], [0, 300, 500, 1000, 20], [[0, 300], [500, 1000]],
                    [0, 300, 500, 1001], [0, -1, 500, 1000], [0, 300, 500.5, 1000],
                    [0, 300, True, 1000], ["0", 300, 500, 1000], [500, 300, 500, 1000],
                    [0, 900, 500, 900], {"x1": 0, "y1": 300, "x2": 500, "y2": 1000}):
            scene = sample_box_scene()
            scene["regions"][0]["box"] = box
            with self.subTest(box=box), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_region_uses_exactly_one_localization_shape(self):
        both = sample_box_scene()
        both["regions"][0]["polygon"] = [[0, 300], [500, 300], [500, 1000], [0, 1000]]
        with self.assertRaises(ValueError):
            parse_scene(json.dumps(both))
        scene = sample_box_scene()
        del scene["regions"][0]["box"]
        with self.assertRaises(ValueError):
            parse_scene(json.dumps(scene))
        for empty in ([], {}, "0,300,500,1000", None):
            for key in ("box", "polygon"):
                scene = sample_box_scene() if key == "box" else sample_scene()
                scene["regions"][0][key] = empty
                with self.subTest(key=key, empty=empty), self.assertRaises(ValueError):
                    parse_scene(json.dumps(scene))

    def test_duplicate_box_regions_rejected(self):
        scene = sample_box_scene()
        duplicate = copy.deepcopy(scene["regions"][0])
        duplicate["region_id"] = "r9"
        scene["regions"].append(duplicate)
        with self.assertRaises(ValueError):
            parse_scene(json.dumps(scene))
        scene["regions"][-1]["role"] = "neighbor_field"
        self.assertEqual(len(parse_scene(json.dumps(scene))["regions"]), 3)

    def test_v5_instruction_offloads_pixel_geometry_to_segmentation(self):
        v5 = INSTRUCTIONS["functional-scene-v5-box"]
        self.assertIn('"box":[x1,y1,x2,y2]', v5)
        self.assertIn("Never write polygon vertices", v5)
        self.assertIn("segmentation model resolves the pixel outline", v5)
        self.assertNotIn("Never write polygon vertices", INSTRUCTIONS["functional-scene-v1"])

    def test_segment_boundaries_canonicalize_to_two_endpoints(self):
        scene = sample_scene()
        scene["boundaries"][0] = {"boundary_id": "b1", "side_a": "r1", "side_b": "r2", "kind": "field_headland",
                                  "segment": [0, 300, 500, 300], "evidence": "Visible row-end line"}
        parsed = parse_scene(json.dumps(scene))
        self.assertEqual(parsed["boundaries"][0]["points"], [[0, 300], [500, 300]])
        self.assertNotIn("segment", json.dumps(parsed))

    def test_flat_quadruple_boundary_encodings_stay_failures(self):
        # The archived v5-box responses wrote boundaries as one flat quadruple. Accepting that
        # shape after the fact would promote those recorded failures, so it stays rejected.
        for points in ([130, 0, 300, 200], [[130, 0, 300, 200]], [[130, 0], [300, 200], [130, 0, 300, 200]]):
            scene = sample_scene()
            scene["boundaries"][0]["points"] = points
            with self.subTest(points=points), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_rejects_invalid_or_mixed_boundary_geometry(self):
        for segment in ([], [0, 300, 500], [0, 300, 500, 300, 1], [[0, 300], [500, 300]], [0, 300, 500, 1001],
                        [0, 300, 500.0, 300], [0, 300, 500, 300][:2] + [0, 300], {"a": 1}, "0,300,500,300"):
            scene = sample_scene()
            scene["boundaries"][0] = {"boundary_id": "b1", "side_a": "r1", "side_b": "r2", "kind": "unknown",
                                      "segment": segment, "evidence": "Visible edge"}
            with self.subTest(segment=segment), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))
        for both in ({"segment": [0, 300, 500, 300]},
                     {"segment": [0, 300, 500, 300], "points": [[0, 300], [500, 300]]}):
            scene = sample_scene()
            scene["boundaries"][0].update(both)
            with self.subTest(both=sorted(both)), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_unknowns_are_validated_rather_than_silently_dropped(self):
        for value in ("", "   ", "x" * 701, None, 3, []):
            scene = sample_scene()
            scene["unknowns"] = [value]
            with self.subTest(value=str(value)[:12]), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))
        scene = {"regions": [], "relations": [], "boundaries": [], "unknowns": ["x" * 701]}
        with self.assertRaises(ValueError) as raised:
            parse_scene(json.dumps(scene))
        self.assertIn("unknown requires", str(raised.exception))

    def test_v6_box_and_segment_contract_matches_the_parser(self):
        v6 = INSTRUCTIONS["functional-scene-v6-box"]
        for phrase in ('"box":[x1,y1,x2,y2]', '"segment":[x1,y1,x2,y2]', "do not write nested point pairs",
                       "between 1 and 700 characters", "return an empty boundaries list"):
            self.assertIn(phrase, v6)
        self.assertIn('"segment":[10,600,500,600]', v6)

    def test_every_region_field_is_required_even_empty_uncertainties(self):
        for key in sample_scene()["regions"][0]:
            scene = sample_scene()
            del scene["regions"][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))
        scene = sample_scene()
        scene["regions"][0]["uncertainties"] = []
        self.assertEqual(parse_scene(json.dumps(scene)), scene)

    def test_duplicate_ids_and_regions_rejected(self):
        for same_id in (True, False):
            scene = sample_scene()
            duplicate = copy.deepcopy(scene["regions"][0])
            if not same_id:
                duplicate["region_id"] = "r3"
            scene["regions"].append(duplicate)
            with self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_bad_relations_rejected(self):
        for changes in ({"subject": "missing"}, {"object": "r2"}, {"relation": "safe_to_turn"},
                        {"subject": "r1", "object": "r2"}, {"relation": "obstacle_in"},
                        {"relation": "access_road_to"}):
            scene = sample_scene()
            scene["relations"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_bad_boundaries_rejected(self):
        for changes in ({"side_a": "missing"}, {"side_b": "r1"}, {"kind": "safe_boundary"},
                        {"points": [[0, 0]]}, {"points": [[0, 0], [0, 0]]}):
            scene = sample_scene()
            scene["boundaries"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_text_and_document_limits(self):
        for text in ("```json\n{}\n```", "prefix {}", "{} {}", "x" * 131073,
                     '{"regions":[],"regions":[],"relations":[],"boundaries":[],"unknowns":["x"]}'):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                parse_scene(text)
        scene = sample_scene()
        scene["regions"][0]["evidence"] = "x" * 701
        with self.assertRaises(ValueError):
            parse_scene(json.dumps(scene))


class SceneRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.frames = []
        for i in range(5):
            data = cv2.imencode(".png", np.full((20, 40, 3), i * 30, np.uint8))[1].tobytes()
            filename = f"image{i}.png"
            (self.source / filename).write_bytes(data)
            self.frames.append({"frame_id": f"f{i}", "frame_index": i, "capture_time_s": i * .5,
                                "video_name": "input.mp4", "video_sha256": "a" * 64,
                                "image_path": filename, "artifact_sha256": {filename: hashlib.sha256(data).hexdigest()}})
        self.save_source()
        self.scene = sample_scene()
        self.response = (json.dumps(self.scene), {"finish_reason": "stop", "model": "fake-provider"})

    def save_source(self):
        write_json(self.source / "manifest.json", {"schema_version": "field_boundary_run_v1", "frames": self.frames})

    def args(self, **changes):
        values = dict(results=self.source, frames=["f2"], output=self.root / "scenes", mode="causal",
                      history_frames=2, history_seconds=2., model="fake", base_url=DEFAULT_BASE_URL,
                      api_key_env=DEFAULT_API_KEY_ENV, max_output_tokens=4096, timeout=30., resume=False,
                      prompt_version=PROMPT_VERSION, response_format="text", continue_on_failure=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_context_excludes_future_foreign_video_and_old_frames(self):
        frames = {f["frame_id"]: f for f in self.frames}
        frames["foreign"] = {**frames["f1"], "frame_id": "foreign", "video_sha256": "b" * 64}
        picked = select_context(frames, "f3", "causal", 2, .75)
        self.assertEqual([f["frame_id"] for f in picked], ["f2", "f3"])
        self.assertEqual(select_context(frames, "f3", "single", 2, 2), [frames["f3"]])

    def test_early_target_uses_only_available_history(self):
        frames = {f["frame_id"]: f for f in self.frames}
        self.assertEqual([f["frame_id"] for f in select_context(frames, "f0", "causal", 2, 2)], ["f0"])

    def test_source_rejects_duplicate_and_noncausal_timestamps(self):
        for modification in ("duplicate", "reversed", "nan"):
            original = copy.deepcopy(self.frames)
            if modification == "duplicate":
                self.frames[1]["frame_index"] = 0
            elif modification == "reversed":
                self.frames[1]["capture_time_s"] = 3
            else:
                self.frames[1]["capture_time_s"] = -1
            self.save_source()
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                source_frames(ResultStore(self.source))
            self.frames = original
        self.save_source()

    def test_causal_run_binds_raw_images_and_no_annotation_input(self):
        self.frames[2]["annotation_reference"] = {"secret_stage": "never-send-this"}
        self.save_source()
        args = self.args()
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            manifest = run(args)
            sent, instruction = model.return_value.ask_images.call_args.args
            self.assertEqual(model.return_value.ask_images.call_args.kwargs,
                             {"response_key": "regions", "response_format": None})
        self.assertIsNone(manifest["configuration"]["response_format"])
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[0][0], (self.source / "image0.png").read_bytes())
        self.assertEqual(sent[2][0], (self.source / "image2.png").read_bytes())
        self.assertNotIn("never-send-this", instruction)
        self.assertIn("image 3: CURRENT", instruction)
        self.assertIn("relative time -1.000s", instruction)
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(all(manifest[key] is False for key in GATES))
        _, prepared = load_scene_run(self.source, args.output)
        self.assertEqual(prepared[0]["scene"], self.scene)
        record = prepared[0]["record"]
        self.assertIsNone(record["response_format"])
        self.assertEqual(record["raw_response"], self.response[0])
        self.assertEqual([f["frame_id"] for f in record["context"]], ["f0", "f1", "f2"])

    def test_json_object_passes_format_and_preserves_archived_response(self):
        args = self.args(frames=["f2", "f3"], response_format="json_object")
        raw = "\n" + json.dumps(self.scene, indent=2) + "\n"
        timing = {"finish_reason": "stop", "model": "fake-provider"}
        response_format = {"type": "json_object"}
        with patch("run_scene_understanding.CloudVLM", autospec=True) as model:
            model.return_value.ask_images.return_value = (raw, timing)
            manifest = run(args)
            calls = model.return_value.ask_images.call_args_list
        self.assertEqual(manifest["configuration"]["response_format"], response_format)
        before = {p.name: p.read_bytes() for p in args.output.iterdir()}
        loaded, items = load_scene_run(self.source, args.output)
        self.assertEqual(loaded, manifest)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(items), 2)
        for call, item in zip(calls, items):
            record = item["record"]
            self.assertEqual(call.kwargs, {"response_key": "regions", "response_format": response_format})
            self.assertEqual(call.args[1], instruction_for(record["context"]))
            self.assertEqual(record["instruction"], call.args[1])
            self.assertEqual(record["response_format"], response_format)
            self.assertEqual(record["raw_response"], raw)
            self.assertEqual(record["timing"], timing)
            self.assertEqual(item["scene"], self.scene)
        self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_legacy_text_archives_load_without_adding_format_or_changing_bytes(self):
        args = self.args()
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            original_manifest = run(args)
        record_path = args.output / "f2.json"
        original_record = json.loads(record_path.read_bytes())
        for missing_config, missing_record in ((True, True), (True, False), (False, True)):
            with self.subTest(missing_config=missing_config, missing_record=missing_record):
                manifest, record = copy.deepcopy(original_manifest), copy.deepcopy(original_record)
                if missing_config:
                    del manifest["configuration"]["response_format"]
                if missing_record:
                    del record["response_format"]
                write_json(record_path, record)
                manifest["frames"][0]["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
                write_json(args.output / "manifest.json", manifest)
                before = {p.name: p.read_bytes() for p in args.output.iterdir()}
                loaded, items = load_scene_run(self.source, args.output)
                self.assertEqual(loaded, manifest)
                self.assertEqual(items[0]["record"], record)
                self.assertEqual(items[0]["scene"], self.scene)
                identity = (record, "f2", record["context"], record["instruction"])
                check_identity(*identity)
                self.assertEqual(check_record(*identity), self.scene)
                for check in (check_identity, check_record):
                    with self.assertRaisesRegex(ValueError, "response format"):
                        check(*identity, response_format={"type": "json_object"})
                self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_single_frame_run_uses_exactly_one_image(self):
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            run(self.args(mode="single"))
            self.assertEqual(len(model.return_value.ask_images.call_args.args[0]), 1)

    def test_each_prompt_version_is_sent_and_loadable_without_changing_archives(self):
        for version, instruction in INSTRUCTIONS.items():
            args = self.args(prompt_version=version, output=self.root / version)
            with self.subTest(version=version), patch("run_scene_understanding.CloudVLM") as model:
                model.return_value.ask_images.return_value = self.response
                manifest = run(args)
                self.assertTrue(model.return_value.ask_images.call_args.args[1].startswith(instruction))
                self.assertEqual(manifest["configuration"]["prompt_version"], version)
            before = {p.name: p.read_bytes() for p in args.output.iterdir()}
            _, items = load_scene_run(self.source, args.output)
            self.assertEqual(items[0]["scene"], self.scene)
            self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_prompt_switch_cannot_resume_or_relabel_existing_run(self):
        args = self.args(prompt_version="functional-scene-v1")
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            run(args)
        args.resume, args.prompt_version = True, "functional-scene-v2"
        with patch("run_scene_understanding.CloudVLM") as model, self.assertRaises(ValueError):
            run(args)
        model.assert_not_called()
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["configuration"]["prompt_version"] = "functional-scene-v2"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "instruction hash mismatch"):
            load_scene_run(self.source, args.output)

    def test_cli_response_format_choices_preserve_text_and_v1_defaults(self):
        argv = ["run_scene_understanding.py", "--results", str(self.source), "--frames", "f2",
                "--output", str(self.args().output), "--mode", "single"]
        for option in (None, "text", "json_object"):
            extra = [] if option is None else ["--response-format", option]
            with self.subTest(option=option), patch.object(sys, "argv", argv + extra), \
                    patch("run_scene_understanding.run") as runner:
                main()
                args = runner.call_args.args[0]
                self.assertEqual(args.response_format, "text" if option is None else option)
                self.assertEqual(args.prompt_version, "functional-scene-v1")
        with patch.object(sys, "argv", argv + ["--response-format", "json_schema"]), \
                patch("sys.stderr"), patch("run_scene_understanding.run") as runner:
            with self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            runner.assert_not_called()

    def test_unknown_options_and_invalid_timeout_stop_before_api(self):
        for changes in ({"prompt_version": "unknown"}, {"timeout": float("nan")},
                        {"timeout": 0}, {"timeout": 3601}, {"response_format": "json_schema"},
                        {"response_format": ""}, {"response_format": None},
                        {"response_format": {"type": "json_object"}}):
            with self.subTest(changes=changes), patch("run_scene_understanding.CloudVLM") as model:
                with self.assertRaises(ValueError):
                    run(self.args(**changes))
                model.assert_not_called()
        self.assertFalse(self.args().output.exists())

    def test_run_requires_response_format_argument(self):
        args = self.args()
        del args.response_format
        with patch("run_scene_understanding.CloudVLM") as model:
            with self.assertRaises(AttributeError):
                run(args)
            model.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_extended_timeout_is_passed_to_cloud_client(self):
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            run(self.args(timeout=1200))
            self.assertEqual(model.call_args.args[-1], 1200)

    def test_resume_skips_api_and_preserves_response(self):
        for option in ("text", "json_object"):
            args = self.args(response_format=option, output=self.root / option)
            with self.subTest(option=option), patch("run_scene_understanding.CloudVLM") as model:
                model.return_value.ask_images.return_value = self.response
                run(args)
            raw_before = (args.output / "f2.json").read_bytes()
            args.resume = True
            with patch("run_scene_understanding.CloudVLM") as model:
                run(args)
                model.assert_not_called()
            self.assertEqual(raw_before, (args.output / "f2.json").read_bytes())

    def test_changed_response_format_cannot_resume_completed_or_failed_run(self):
        for option in ("text", "json_object"):
            for failed in (False, True):
                args = self.args(response_format=option, output=self.root / f"{option}-{failed}")
                with self.subTest(option=option, failed=failed), patch("run_scene_understanding.CloudVLM") as model:
                    if failed:
                        model.return_value.ask_images.return_value = ("not json", {"finish_reason": "stop"})
                        with self.assertRaises(ValueError):
                            run(args)
                    else:
                        model.return_value.ask_images.return_value = self.response
                        run(args)
                before = {p.name: p.read_bytes() for p in args.output.iterdir()}
                args.resume = True
                args.response_format = "json_object" if option == "text" else "text"
                with patch("run_scene_understanding.CloudVLM") as model:
                    with self.assertRaisesRegex(ValueError, "Resume configuration mismatch"):
                        run(args)
                    model.assert_not_called()
                self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_response_format_config_or_record_relabeling_is_rejected(self):
        for option, other_format in (("text", {"type": "json_object"}), ("json_object", None)):
            args = self.args(response_format=option, output=self.root / option)
            with patch("run_scene_understanding.CloudVLM") as model:
                model.return_value.ask_images.return_value = self.response
                original_manifest = run(args)
            record_path = args.output / "f2.json"
            original_record = json.loads(record_path.read_bytes())
            args.resume = True
            for target, remove in (("configuration", False), ("record", False),
                                   ("configuration", True), ("record", True)):
                if remove and option == "text":
                    continue
                with self.subTest(option=option, target=target, remove=remove):
                    manifest, record = copy.deepcopy(original_manifest), copy.deepcopy(original_record)
                    changed = manifest["configuration"] if target == "configuration" else record
                    if remove:
                        del changed["response_format"]
                    else:
                        changed["response_format"] = other_format
                    write_json(record_path, record)
                    manifest["frames"][0]["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
                    write_json(args.output / "manifest.json", manifest)
                    before = {p.name: p.read_bytes() for p in args.output.iterdir()}
                    with self.assertRaisesRegex(ValueError, "response format"):
                        load_scene_run(self.source, args.output)
                    with patch("run_scene_understanding.CloudVLM") as model:
                        with self.assertRaises(ValueError):
                            run(args)
                        model.assert_not_called()
                    self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_load_rejects_unsupported_format_even_when_record_and_config_agree(self):
        args = self.args()
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            manifest = run(args)
        record_path = args.output / "f2.json"
        record = json.loads(record_path.read_bytes())
        for value in ("text", "json_object", {}, {"type": "text"}, {"type": "json_schema"},
                      {"type": "json_object", "extra": True}, [], False, 0):
            with self.subTest(value=value):
                manifest["configuration"]["response_format"] = value
                record["response_format"] = value
                write_json(record_path, record)
                manifest["frames"][0]["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
                write_json(args.output / "manifest.json", manifest)
                before = {p.name: p.read_bytes() for p in args.output.iterdir()}
                with self.assertRaisesRegex(ValueError, "Unsupported scene response format"):
                    load_scene_run(self.source, args.output)
                self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_interrupted_batch_recovers_orphan_successful_record(self):
        args = self.args(frames=["f2", "f3"])
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.side_effect = [self.response, KeyboardInterrupt()]
            with self.assertRaises(KeyboardInterrupt):
                run(args)
        args.resume = True
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            manifest = run(args)
            self.assertEqual(model.return_value.ask_images.call_count, 1)
        self.assertEqual(len(manifest["frames"]), 2)

    def test_resume_refuses_changed_config_and_tampered_response(self):
        args = self.args()
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            run(args)
        args.resume = True
        args.mode = "single"
        with self.assertRaises(ValueError):
            run(args)
        args.mode = "causal"
        record = json.loads((args.output / "f2.json").read_bytes())
        record["scene"]["unknowns"] = ["tampered"]
        write_json(args.output / "f2.json", record)
        with self.assertRaises(ValueError):
            run(args)
        with self.assertRaises(ValueError):
            load_scene_run(self.source, args.output)

    def test_truncated_and_invalid_responses_are_archived_not_empty_success(self):
        invalid_empty = json.dumps({"regions": [], "relations": [], "boundaries": [], "unknowns": []})
        responses = (("length", self.response[0]), ("stop", "not json"), (None, self.response[0]),
                     ("stop", invalid_empty), ("stop", "```json\n" + self.response[0] + "\n```"))
        for option, response_format in (("text", None), ("json_object", {"type": "json_object"})):
            for index, (reason, response) in enumerate(responses):
                output = self.root / f"bad-{option}-{index}"
                with self.subTest(option=option, reason=reason, index=index), \
                        patch("run_scene_understanding.CloudVLM") as model:
                    model.return_value.ask_images.return_value = (response, {"finish_reason": reason})
                    with self.assertRaises(ValueError):
                        run(self.args(output=output, response_format=option))
                record = json.loads((output / "f2.json").read_bytes())
                self.assertEqual(record["raw_response"], response)
                self.assertEqual(record["timing"], {"finish_reason": reason})
                self.assertEqual(record["response_format"], response_format)
                self.assertEqual(record["status"], "error")
                self.assertNotIn("scene", record)
                manifest = json.loads((output / "manifest.json").read_bytes())
                self.assertEqual(manifest["status"], "incomplete")
                self.assertEqual(manifest["configuration"]["response_format"], response_format)
                with self.assertRaises(ValueError):
                    load_scene_run(self.source, output)
                with self.assertRaises(ValueError):
                    check_record(record, "f2", record["context"], record["instruction"], response_format)

    def test_continue_on_failure_gives_every_frame_one_request_and_drops_failures(self):
        output = self.root / "partial"
        responses = [self.response, ("not json", {"finish_reason": "stop"}), self.response]
        with patch("run_scene_understanding.CloudVLM", autospec=True) as model:
            model.return_value.ask_images.side_effect = responses
            manifest = run(self.args(frames=["f0", "f1", "f2"], output=output, continue_on_failure=True))
        self.assertEqual(model.return_value.ask_images.call_count, 3)
        self.assertEqual(manifest["status"], "completed_with_failures")
        self.assertEqual(manifest["failed_frame_ids"], ["f1"])
        self.assertEqual([entry["frame_id"] for entry in manifest["frames"]], ["f0", "f1", "f2"])
        failure = json.loads((output / "f1.json").read_bytes())
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["raw_response"], "not json")
        self.assertNotIn("scene", failure)
        loaded, items = load_scene_run(self.source, output)
        self.assertEqual(loaded, manifest)
        self.assertEqual([item["frame"]["frame_id"] for item in items], ["f0", "f2"])
        self.assertTrue(all(item["scene"] == self.scene for item in items))

    def test_continue_on_failure_refuses_resume_and_a_run_without_any_scene_refuses_load(self):
        with patch("run_scene_understanding.CloudVLM") as model:
            with self.assertRaises(ValueError):
                run(self.args(continue_on_failure=True, resume=True))
            model.assert_not_called()
        output = self.root / "all-failed"
        with patch("run_scene_understanding.CloudVLM", autospec=True) as model:
            model.return_value.ask_images.side_effect = [("not json", {"finish_reason": "stop"})] * 2
            manifest = run(self.args(frames=["f0", "f1"], output=output, continue_on_failure=True))
        self.assertEqual(manifest["failed_frame_ids"], ["f0", "f1"])
        with self.assertRaises(ValueError):
            load_scene_run(self.source, output)

    def test_completed_with_failures_accounting_is_replayed_on_load(self):
        output = self.root / "accounting"
        with patch("run_scene_understanding.CloudVLM", autospec=True) as model:
            model.return_value.ask_images.side_effect = [self.response, ("not json", {"finish_reason": "stop"})]
            manifest = run(self.args(frames=["f0", "f1"], output=output, continue_on_failure=True))
        for changes in ({"status": "completed"}, {"failed_frame_ids": []}, {"failed_frame_ids": ["f0"]}, {}):
            tampered = copy.deepcopy(manifest)
            tampered.update(changes)
            if not changes:
                tampered.pop("failed_frame_ids")
            write_json(output / "manifest.json", tampered)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_scene_run(self.source, output)
        record = json.loads((output / "f1.json").read_bytes())
        record["status"] = "response_received"
        write_json(output / "f1.json", record)
        manifest["frames"][1]["sha256"] = file_hash(output / "f1.json")
        write_json(output / "manifest.json", manifest)
        with self.assertRaises(ValueError):
            load_scene_run(self.source, output)

    def test_json_object_bbox_response_is_archived_failure_not_repaired(self):
        scene = sample_scene()
        scene["regions"][0]["polygon"] = [[0, 300, 500, 1000]]
        raw = json.dumps(scene)
        args = self.args(response_format="json_object")
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = (raw, {"finish_reason": "stop"})
            with self.assertRaises(ValueError):
                run(args)
        record = json.loads((args.output / "f2.json").read_bytes())
        self.assertEqual(record["raw_response"], raw)
        self.assertEqual(record["response_format"], {"type": "json_object"})
        self.assertEqual(record["status"], "error")
        self.assertNotIn("scene", record)
        before = {p.name: p.read_bytes() for p in args.output.iterdir()}
        with self.assertRaises(ValueError):
            load_scene_run(self.source, args.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in args.output.iterdir()})

    def test_empty_scene_is_valid_success(self):
        scene = {"regions": [], "relations": [], "boundaries": [], "unknowns": ["Cannot see field"]}
        for option in ("text", "json_object"):
            args = self.args(response_format=option, output=self.root / option)
            with self.subTest(option=option), patch("run_scene_understanding.CloudVLM") as model:
                model.return_value.ask_images.return_value = (json.dumps(scene), {"finish_reason": "stop"})
                run(args)
            _, items = load_scene_run(self.source, args.output)
            self.assertEqual(items[0]["scene"], scene)

    def test_source_hash_failure_stops_before_api(self):
        (self.source / "image1.png").write_bytes(b"changed")
        with patch("run_scene_understanding.CloudVLM") as model:
            with self.assertRaises(ValueError):
                run(self.args())
            model.assert_not_called()

    def test_missing_key_does_not_create_output_or_send_request(self):
        args = self.args()
        with patch.dict("os.environ", {}, clear=True), patch("urllib.request.urlopen") as send:
            with self.assertRaises(ValueError):
                run(args)
            send.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_resume_rejects_promoted_safety_gates(self):
        args = self.args()
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            run(args)
        manifest = json.loads((args.output / "manifest.json").read_bytes())
        manifest["maneuver_verified"] = True
        write_json(args.output / "manifest.json", manifest)
        args.resume = True
        with self.assertRaises(ValueError):
            run(args)

    def test_does_not_overwrite_existing_output(self):
        args = self.args()
        args.output.mkdir()
        (args.output / "sentinel").write_bytes(b"keep")
        with self.assertRaises(ValueError):
            run(args)
        self.assertEqual((args.output / "sentinel").read_bytes(), b"keep")

    def test_rejects_unapproved_endpoint_without_api(self):
        with patch("run_scene_understanding.CloudVLM") as model:
            with self.assertRaises(ValueError):
                run(self.args(base_url="https://provider.example/v1"))
            model.assert_not_called()

    def test_does_not_write_inside_source_run(self):
        output = self.source / "new-scenes"
        with patch("run_scene_understanding.CloudVLM") as model:
            with self.assertRaises(ValueError):
                run(self.args(output=output))
            model.assert_not_called()
        self.assertFalse(output.exists())

    def test_network_failure_is_not_archived_as_valid_empty_scene(self):
        args = self.args()
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.side_effect = RuntimeError("provider failure")
            with self.assertRaises(RuntimeError):
                run(args)
        record = json.loads((args.output / "f2.json").read_bytes())
        self.assertEqual(record["status"], "error")
        self.assertNotIn("scene", record)
        self.assertNotIn("provider failure", (args.output / "f2.json").read_text())

    def test_resume_retries_failed_frame_without_overwriting_failed_evidence(self):
        args = self.args(frames=["f2", "f3"])
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.side_effect = [self.response, ("truncated", {"finish_reason": "length"})]
            with self.assertRaises(ValueError):
                run(args)
        before = (args.output / "f3.json").read_bytes()
        args.resume = True
        with patch("run_scene_understanding.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response
            manifest = run(args)
            self.assertEqual(model.return_value.ask_images.call_count, 1)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual((args.output / "f3.attempt-0001.json").read_bytes(), before)
        self.assertEqual(manifest["failed_attempts"][0]["sha256"], hashlib.sha256(before).hexdigest())
        _, items = load_scene_run(self.source, args.output)
        self.assertEqual(len(items), 2)
        with patch("run_scene_understanding.CloudVLM") as model:
            run(args)
            model.assert_not_called()
        (args.output / "f3.attempt-0001.json").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            run(args)


if __name__ == "__main__":
    unittest.main()
