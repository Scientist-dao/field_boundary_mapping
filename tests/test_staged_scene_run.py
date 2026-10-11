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
from run_offline import file_hash, write_json
from run_scene_understanding import load_scene_run
from run_staged_scene import INVENTORY_VERSION, STAGED_VERSION, RESPONSE_FORMAT, load_staged_scene_run, main, run
from scene_understanding import GATES, scene_queries


class StagedSceneRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.frames = []
        for i in range(4):
            data = cv2.imencode(".png", np.full((20, 40, 3), 30 * i, np.uint8))[1].tobytes()
            filename = f"image{i}.png"
            (self.source / filename).write_bytes(data)
            self.frames.append({"frame_id": f"f{i}", "frame_index": i, "capture_time_s": i * .5,
                                "video_name": "input.mp4", "video_sha256": "a" * 64,
                                "image_path": filename, "annotation_reference": {"stage": "DO_NOT_SEND"},
                                "artifact_sha256": {filename: hashlib.sha256(data).hexdigest()}})
        write_json(self.source / "manifest.json", {"schema_version": "field_boundary_run_v1", "frames": self.frames})
        self.catalogue = {"regions": [
            {"region_id": "r1", "role": "current_field", "surface": "crop", "description": "crop patch on left",
             "evidence": "visible working rows", "uncertainties": ["ownership unknown"]},
            {"region_id": "r2", "role": "headland_candidate", "surface": "ground", "description": "strip above crops",
             "evidence": "row ends and cross strip", "uncertainties": ["clearance unknown"]}],
            "relations": [{"subject": "r2", "relation": "headland_of", "object": "r1", "evidence": "strip at row ends"}],
            "boundaries": [{"boundary_id": "b1", "side_a": "r1", "side_b": "r2", "kind": "field_headland",
                            "description": "crop to bare strip edge", "evidence": "visible row end edge"}],
            "unknowns": ["turning permission unknown"]}
        self.localized = {"regions": [
            {"region_id": "r1", "polygon": [[0, 300], [500, 300], [500, 1000], [0, 1000]], "reason": "visible crop patch"},
            {"region_id": "r2", "polygon": [[0, 200], [500, 200], [500, 300], [0, 300]], "reason": "visible strip"}],
            "boundaries": [{"boundary_id": "b1", "points": [[0, 300], [500, 300]], "reason": "visible edge"}]}

    def args(self, **changes):
        values = dict(results=self.source, frames=["f2"], output=self.root / "staged", mode="causal",
                      history_frames=2, history_seconds=2.0, model="fake", base_url=DEFAULT_BASE_URL,
                      api_key_env=DEFAULT_API_KEY_ENV, max_output_tokens=4096, timeout=30.0, pipeline_version=STAGED_VERSION)
        values.update(changes)
        return argparse.Namespace(**values)

    def response(self, value, reason="stop"):
        return json.dumps(value), {"finish_reason": reason, "model": "fake", "parsed_from": "content", "strict_content": True}

    def inventory_documents(self):
        inventory = {"items": [
            {"item_id": r["region_id"], "surface": r["surface"], "description": r["description"]}
            for r in self.catalogue["regions"]], "unknowns": ["Inventory uncertainty"]}
        assignments = {"assignments": [
            {"item_id": r["region_id"], **{k: r[k] for k in ("role", "evidence", "uncertainties")}}
            for r in self.catalogue["regions"]],
            **{k: copy.deepcopy(self.catalogue[k]) for k in ("relations", "boundaries", "unknowns")}}
        return inventory, assignments

    def run_valid(self, pipeline_version=STAGED_VERSION, **changes):
        documents = [self.catalogue, self.localized] if pipeline_version == STAGED_VERSION else [
            *self.inventory_documents(), self.localized]
        with patch("run_staged_scene.CloudVLM", autospec=True) as model:
            model.return_value.ask_images.side_effect = [self.response(document) for document in documents]
            manifest = run(self.args(pipeline_version=pipeline_version, **changes))
            calls = model.return_value.ask_images.call_args_list
        return manifest, calls

    def test_causal_semantics_and_current_only_localization_are_bound(self):
        manifest, calls = self.run_valid()
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0].args[0]), 3)
        self.assertEqual(len(calls[1].args[0]), 1)
        self.assertEqual(calls[1].args[0][0][0], (self.source / "image2.png").read_bytes())
        self.assertIn("image 3: CURRENT", calls[0].args[1])
        for call in calls:
            self.assertNotIn("DO_NOT_SEND", call.args[1])
            self.assertEqual(call.kwargs["response_format"], RESPONSE_FORMAT)
            self.assertIs(call.kwargs["strict_content"], True)
        self.assertNotIn("row ends and cross strip", calls[1].args[1])
        before = {p.name: p.read_bytes() for p in self.args().output.iterdir()}
        loaded, items = load_scene_run(self.source, self.args().output)
        self.assertEqual(loaded, manifest)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.args().output.iterdir()})
        record = items[0]["record"]
        self.assertNotIn("raw_response", record)
        for name, expected in (("catalogue", self.catalogue), ("localization", self.localized)):
            self.assertEqual(record["stages"][name]["raw_response"], json.dumps(expected))
        self.assertEqual(record["scene"]["regions"][1]["role"], "headland_candidate")
        self.assertEqual(len(scene_queries(record["scene"], 40, 20)), 2)
        for obj in (manifest, record, *record["stages"].values(), record["localization_audit"]):
            self.assertTrue(all(obj[key] is False for key in GATES))

    def test_staged_archive_feeds_existing_offline_refinement_with_bound_provenance(self):
        import refine_scene_regions as refinement
        self.run_valid()
        before = {p.name: p.read_bytes() for p in self.args().output.iterdir()}
        weights = self.root / "synthetic.pt"
        weights.write_bytes(b"SYNTHETIC TEST ONLY")
        args = argparse.Namespace(results=self.source, scenes=self.args().output, output=self.root / "refined",
                                  weights=weights, clip_dir=None, confidence=0.3, device="cuda", threads=4)
        with patch.object(refinement, "LocalSAM3") as model:
            model.return_value.metadata = {}
            model.return_value.predict_regions.return_value = ([], {"total_seconds": 0.0})
            manifest = refinement.run(args)
            self.assertEqual(model.return_value.predict_regions.call_count, 1)
        self.assertEqual(manifest["status"], "completed")
        hashes = manifest["configuration"]["implementation_sha256"]
        self.assertIn("scene_staging.py", hashes)
        self.assertIn("run_staged_scene.py", hashes)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.args().output.iterdir()})
        self.assertEqual(len(manifest["frames"][0]["artifact_sha256"]), 8)
        for relative, digest in manifest["frames"][0]["artifact_sha256"].items():
            self.assertEqual(file_hash(args.output / relative), digest)

    def test_single_mode_never_sends_history_to_either_stage(self):
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.side_effect = [self.response(self.catalogue), self.response(self.localized)]
            run(self.args(mode="single"))
        self.assertTrue(all(len(call.args[0]) == 1 for call in model.return_value.ask_images.call_args_list))

    def test_catalogue_failure_preserves_raw_and_prevents_localization(self):
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.return_value = ('{"regions": []}', {"finish_reason": "stop"})
            with self.assertRaises(ValueError):
                run(self.args())
        self.assertEqual(model.return_value.ask_images.call_count, 1)
        record = json.loads((self.args().output / "f2.json").read_bytes())
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["stages"]["catalogue"]["raw_response"], '{"regions": []}')
        self.assertNotIn("localization", record["stages"])
        with self.assertRaises(ValueError):
            load_staged_scene_run(self.source, self.args().output)

    def test_bbox_localization_is_not_repaired_and_catalogue_is_preserved(self):
        self.localized["regions"][0]["polygon"] = [[0, 300, 500, 1000]]
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.side_effect = [self.response(self.catalogue), self.response(self.localized)]
            with self.assertRaises(ValueError):
                run(self.args())
        record = json.loads((self.args().output / "f2.json").read_bytes())
        self.assertEqual(record["stages"]["catalogue"]["status"], "parsed")
        self.assertEqual(record["stages"]["localization"]["raw_response"], json.dumps(self.localized))
        self.assertNotIn("scene", record)
        manifest = json.loads((self.args().output / "manifest.json").read_bytes())
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(manifest["frames"][0]["sha256"], file_hash(self.args().output / "f2.json"))

    def test_truncation_in_either_stage_is_archived_failure(self):
        for stage_index in (0, 1):
            output = self.root / str(stage_index)
            responses = [self.response(self.catalogue), self.response(self.localized)]
            raw, timing = responses[stage_index]
            responses[stage_index] = (raw, {**timing, "finish_reason": "length"})
            with self.subTest(stage=stage_index), patch("run_staged_scene.CloudVLM") as model:
                model.return_value.ask_images.side_effect = responses
                with self.assertRaises(ValueError):
                    run(self.args(output=output))
            record = json.loads((output / "f2.json").read_bytes())
            stage = record["stages"][("catalogue", "localization")[stage_index]]
            self.assertEqual(stage["raw_response"], raw)
            self.assertEqual(record["status"], "error")
            self.assertEqual(model.return_value.ask_images.call_count, stage_index + 1)

    def test_nontext_final_content_is_archived_and_reasoning_cannot_rescue_it(self):
        for index, raw in enumerate((None, self.catalogue, "invalid final text")):
            output = self.root / f"nontext-{index}"
            timing = {"finish_reason": "stop", "parsed_from": "content", "strict_content": True,
                      "reasoning_content": json.dumps(self.catalogue)}
            with patch("run_staged_scene.CloudVLM") as model:
                model.return_value.ask_images.return_value = (raw, timing)
                with self.assertRaises(ValueError):
                    run(self.args(output=output))
                self.assertEqual(model.return_value.ask_images.call_count, 1)
            record = json.loads((output / "f2.json").read_bytes())
            self.assertEqual(record["stages"]["catalogue"]["raw_response"], raw)
            self.assertNotIn("scene", record)

    def test_empty_catalogue_explicitly_skips_second_call(self):
        empty = {"regions": [], "relations": [], "boundaries": [], "unknowns": ["cannot see objects"]}
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response(empty)
            run(self.args())
        self.assertEqual(model.return_value.ask_images.call_count, 1)
        _, items = load_scene_run(self.source, self.args().output)
        second = items[0]["record"]["stages"]["localization"]
        self.assertEqual(second["status"], "skipped_no_targets")
        self.assertNotIn("raw_response", second)
        self.assertEqual(items[0]["scene"], empty)

    def test_new_output_only_and_invalid_inputs_do_not_call_cloud(self):
        self.args().output.mkdir()
        sentinel = self.args().output / "keep"
        sentinel.write_bytes(b"unchanged")
        cases = [{}, {"output": self.source / "inside"}, {"base_url": "https://unapproved.example"},
                 {"api_key_env": "OTHER_KEY"}, {"timeout": float("nan")}, {"timeout": 3601},
                 {"max_output_tokens": 0}, {"frames": ["f2", "f2"]}, {"frames": ["f9"]}]
        for changes in cases:
            with self.subTest(changes=changes), patch("run_staged_scene.CloudVLM") as model:
                with self.assertRaises(ValueError):
                    run(self.args(**changes))
                model.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"unchanged")

    def test_tampering_derived_scene_or_stage_binding_is_rejected_even_rehashed(self):
        manifest, _ = self.run_valid()
        path = self.args().output / "f2.json"
        original = json.loads(path.read_bytes())
        changes = [lambda r: r["scene"]["regions"][0].update(role="unknown"),
                   lambda r: r["stages"]["localization"].update(context=r["context"]),
                   lambda r: r["stages"]["catalogue"].update(instruction="changed"),
                   lambda r: r["stages"]["localization"].update(response_format=None),
                   lambda r: r["stages"]["localization"]["timing"].update(parsed_from="reasoning_content"),
                   lambda r: r["stages"]["localization"]["timing"].update(strict_content=False),
                   lambda r: r.update(maneuver_verified=True),
                   lambda r: r["localization_audit"].update(semantic_verified=True)]
        for change in changes:
            record = copy.deepcopy(original)
            change(record)
            write_json(path, record)
            manifest["frames"][0]["sha256"] = file_hash(path)
            write_json(self.args().output / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                load_scene_run(self.source, self.args().output)

    def test_tampering_sources_templates_or_order_is_rejected(self):
        manifest, _ = self.run_valid()
        for change in (lambda m: m["configuration"]["template_sha256"].update(catalogue="0" * 64),
                       lambda m: m["configuration"].update(source_manifest_sha256="0" * 64),
                       lambda m: m["configuration"].update(target_frames=["f1"]),
                       lambda m: m.update(physical_field_boundary_verified=True)):
            modified = copy.deepcopy(manifest)
            change(modified)
            write_json(self.args().output / "manifest.json", modified)
            with self.assertRaises(ValueError):
                load_staged_scene_run(self.source, self.args().output)

    def test_missing_key_creates_no_output_and_network_errors_do_not_leak_details(self):
        with patch.dict("os.environ", {}, clear=True), patch("urllib.request.urlopen") as send:
            with self.assertRaises(ValueError):
                run(self.args())
            send.assert_not_called()
        self.assertFalse(self.args().output.exists())
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.side_effect = RuntimeError("DO_NOT_ARCHIVE_PROVIDER_DETAIL")
            with self.assertRaises(RuntimeError):
                run(self.args())
        self.assertNotIn("DO_NOT_ARCHIVE_PROVIDER_DETAIL", (self.args().output / "f2.json").read_text())

    def test_inventory_causal_stages_preserve_visible_items_and_original_responses(self):
        manifest, calls = self.run_valid(INVENTORY_VERSION)
        self.assertEqual([len(c.args[0]) for c in calls], [1, 3, 1])
        self.assertEqual([c.kwargs["response_key"] for c in calls], ["items", "assignments", "regions"])
        for call in calls:
            self.assertEqual(call.kwargs["response_format"], RESPONSE_FORMAT)
            self.assertIs(call.kwargs["strict_content"], True)
            self.assertNotIn("DO_NOT_SEND", call.args[1])
        for index in (0, 2):
            self.assertEqual(calls[index].args[0][0][0], (self.source / "image2.png").read_bytes())
        self.assertIn("image 3: CURRENT", calls[1].args[1])
        self.assertNotIn("Inventory uncertainty", calls[2].args[1])
        self.assertNotIn("visible working rows", calls[2].args[1])
        before = {p.name: p.read_bytes() for p in self.args().output.iterdir()}
        loaded, items = load_scene_run(self.source, self.args().output)
        self.assertEqual(loaded, manifest)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.args().output.iterdir()})
        record = items[0]["record"]
        inventory, assignments = self.inventory_documents()
        for name, value in zip(("inventory", "catalogue", "localization"), (inventory, assignments, self.localized)):
            self.assertEqual(record["stages"][name]["raw_response"], json.dumps(value))
            self.assertEqual(record["stages"][name]["parsed"], value)
        self.assertEqual(record["catalogue"]["regions"], self.catalogue["regions"])
        self.assertEqual(record["catalogue"]["unknowns"], inventory["unknowns"] + assignments["unknowns"])
        for obj in (manifest, record, *record["stages"].values(), record["localization_audit"]):
            self.assertTrue(all(obj[k] is False for k in GATES))

    def test_inventory_single_mode_all_stages_send_current_only(self):
        _, calls = self.run_valid(INVENTORY_VERSION, mode="single")
        self.assertEqual([len(c.args[0]) for c in calls], [1, 1, 1])

    def test_unknown_assignments_keep_all_items_and_sam_queries(self):
        for region in self.catalogue["regions"]:
            region["role"] = "unknown"
        self.catalogue["relations"] = []
        self.run_valid(INVENTORY_VERSION)
        _, items = load_scene_run(self.source, self.args().output)
        scene = items[0]["scene"]
        self.assertEqual([r["region_id"] for r in scene["regions"]], ["r1", "r2"])
        self.assertTrue(all(r["role"] == "unknown" for r in scene["regions"]))
        self.assertEqual(len(scene_queries(scene, 40, 20)), 2)

    def test_invalid_assignment_ids_stop_before_localization_without_repairs(self):
        for operation in ("missing", "new", "renamed", "duplicate"):
            inventory, assignments = self.inventory_documents()
            entries = assignments["assignments"]
            if operation == "missing":
                entries.pop()
            elif operation == "renamed":
                entries[0]["item_id"] = "renamed"
            else:
                entries.append({**entries[0], "item_id": "new" if operation == "new" else "r1"})
            output = self.root / operation
            with self.subTest(operation=operation), patch("run_staged_scene.CloudVLM") as model:
                model.return_value.ask_images.side_effect = [self.response(inventory), self.response(assignments)]
                with self.assertRaises(ValueError):
                    run(self.args(pipeline_version=INVENTORY_VERSION, output=output))
                self.assertEqual(model.return_value.ask_images.call_count, 2)
            record = json.loads((output / "f2.json").read_bytes())
            self.assertEqual(record["stages"]["inventory"]["parsed"], inventory)
            self.assertEqual(record["stages"]["catalogue"]["raw_response"], json.dumps(assignments))
            self.assertNotIn("localization", record["stages"])
            self.assertNotIn("catalogue", record)
            self.assertNotIn("scene", record)
            with self.assertRaises(ValueError):
                load_scene_run(self.source, output)

    def test_empty_inventory_explicitly_skips_both_remaining_calls(self):
        inventory = {"items": [], "unknowns": ["Image fully obscured"]}
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.return_value = self.response(inventory)
            run(self.args(pipeline_version=INVENTORY_VERSION))
            self.assertEqual(model.return_value.ask_images.call_count, 1)
        _, items = load_scene_run(self.source, self.args().output)
        record = items[0]["record"]
        for name, status in (("catalogue", "skipped_no_inventory"), ("localization", "skipped_no_targets")):
            self.assertEqual(record["stages"][name]["status"], status)
            self.assertNotIn("raw_response", record["stages"][name])
            self.assertNotIn("timing", record["stages"][name])
        self.assertEqual(record["scene"], {"regions": [], "relations": [], "boundaries": [],
                                          "unknowns": inventory["unknowns"]})

    def test_inventory_truncation_at_each_stage_is_archived_failure(self):
        for index in range(3):
            responses = [self.response(d) for d in (*self.inventory_documents(), self.localized)]
            raw, timing = responses[index]
            responses[index] = raw, {**timing, "finish_reason": "length"}
            output = self.root / f"truncated-{index}"
            with self.subTest(stage=index), patch("run_staged_scene.CloudVLM") as model:
                model.return_value.ask_images.side_effect = responses
                with self.assertRaises(ValueError):
                    run(self.args(pipeline_version=INVENTORY_VERSION, output=output))
                self.assertEqual(model.return_value.ask_images.call_count, index + 1)
            record = json.loads((output / "f2.json").read_bytes())
            stage = record["stages"][("inventory", "catalogue", "localization")[index]]
            self.assertEqual(stage["raw_response"], raw)
            self.assertEqual(stage["status"], "error")
            self.assertNotIn("scene", record)

    def test_inventory_tampering_rejected_even_with_updated_record_hash(self):
        manifest, _ = self.run_valid(INVENTORY_VERSION)
        path = self.args().output / "f2.json"
        original = json.loads(path.read_bytes())
        changes = [lambda r: r.update(pipeline_version=STAGED_VERSION),
                   lambda r: r["stages"]["inventory"].update(context=r["context"]),
                   lambda r: r["stages"]["inventory"]["parsed"]["items"][0].update(surface="water"),
                   lambda r: r["catalogue"]["regions"][0].update(description="different appearance"),
                   lambda r: r["stages"]["catalogue"]["parsed"]["assignments"].pop(),
                   lambda r: r["stages"].pop("inventory"),
                   lambda r: r["stages"]["inventory"]["timing"].update(parsed_from="reasoning_content")]
        for change in changes:
            record = copy.deepcopy(original)
            change(record)
            write_json(path, record)
            manifest["frames"][0]["sha256"] = file_hash(path)
            write_json(self.args().output / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                load_scene_run(self.source, self.args().output)

    def test_pipeline_cannot_be_relabelled_by_changing_configuration(self):
        from run_staged_scene import template_hashes
        for version, other in ((STAGED_VERSION, INVENTORY_VERSION), (INVENTORY_VERSION, STAGED_VERSION)):
            output = self.root / version
            manifest, _ = self.run_valid(version, output=output)
            manifest["configuration"].update(pipeline_version=other, template_sha256=template_hashes(other))
            write_json(output / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                load_scene_run(self.source, output)

    def test_inventory_archive_feeds_sam_refinement_without_changing_sources(self):
        import refine_scene_regions as refinement
        self.run_valid(INVENTORY_VERSION)
        before = {p.name: p.read_bytes() for p in self.args().output.iterdir()}
        weights = self.root / "synthetic.pt"
        weights.write_bytes(b"SYNTHETIC TEST ONLY")
        args = argparse.Namespace(results=self.source, scenes=self.args().output, output=self.root / "refined",
                                  weights=weights, clip_dir=None, confidence=0.3, device="cuda", threads=4)
        with patch.object(refinement, "LocalSAM3") as model:
            model.return_value.metadata = {}
            model.return_value.predict_regions.return_value = ([], {"total_seconds": 0.0})
            manifest = refinement.run(args)
            self.assertEqual(model.return_value.predict_regions.call_count, 1)
        self.assertEqual(manifest["status"], "completed")
        self.assertIn("scene_staging.py", manifest["configuration"]["implementation_sha256"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.args().output.iterdir()})
        self.assertEqual(len(manifest["frames"][0]["artifact_sha256"]), 8)
        for relative, digest in manifest["frames"][0]["artifact_sha256"].items():
            self.assertEqual(file_hash(args.output / relative), digest)

    def test_cli_keeps_old_default_and_exposes_inventory_only_by_selection(self):
        required = ["run_staged_scene.py", "--results", "source", "--frames", "f2", "--output", "out", "--mode", "single"]
        for extra, expected in (([], STAGED_VERSION), (["--pipeline-version", INVENTORY_VERSION], INVENTORY_VERSION)):
            with patch.object(sys, "argv", required + extra), patch("run_staged_scene.run") as runner:
                main()
                self.assertEqual(runner.call_args.args[0].pipeline_version, expected)
        with patch("run_staged_scene.CloudVLM") as model:
            with self.assertRaises(ValueError):
                run(self.args(pipeline_version="unsupported"))
            model.assert_not_called()

    def test_inventory_assignment_missing_uncertainties_is_not_filled_in(self):
        inventory, assignments = self.inventory_documents()
        for assignment in assignments["assignments"]:
            del assignment["uncertainties"]
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.side_effect = [self.response(inventory), self.response(assignments)]
            with self.assertRaisesRegex(ValueError, "uncertainties"):
                run(self.args(pipeline_version=INVENTORY_VERSION))
            self.assertEqual(model.return_value.ask_images.call_count, 2)
        record = json.loads((self.args().output / "f2.json").read_bytes())
        self.assertEqual(record["stages"]["inventory"]["parsed"], inventory)
        self.assertEqual(record["stages"]["catalogue"]["raw_response"], json.dumps(assignments))
        self.assertNotIn("parsed", record["stages"]["catalogue"])
        self.assertNotIn("catalogue", record)
        self.assertNotIn("localization", record["stages"])

    def test_inventory_localization_bbox_does_not_become_a_polygon(self):
        for region in self.localized["regions"]:
            region["polygon"] = [[0, 300, 500, 1000]]
        with patch("run_staged_scene.CloudVLM") as model:
            model.return_value.ask_images.side_effect = [
                self.response(d) for d in (*self.inventory_documents(), self.localized)]
            with self.assertRaises(ValueError):
                run(self.args(pipeline_version=INVENTORY_VERSION))
            self.assertEqual(model.return_value.ask_images.call_count, 3)
        record = json.loads((self.args().output / "f2.json").read_bytes())
        self.assertEqual(record["catalogue"]["regions"], self.catalogue["regions"])
        self.assertEqual(record["stages"]["localization"]["raw_response"], json.dumps(self.localized))
        self.assertNotIn("parsed", record["stages"]["localization"])
        self.assertNotIn("scene", record)
        with self.assertRaises(ValueError):
            load_scene_run(self.source, self.args().output)


if __name__ == "__main__":
    unittest.main()
