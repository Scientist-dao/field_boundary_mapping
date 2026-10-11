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
from rerender_overlays import add_proposal_passed_variants, file_hash as rerender_file_hash
from review_results import ResultStore
from run_proposals import assess_predictions, cap_verified_detections, prepare_inputs, render_candidates, run
from sam3_local import LocalSAM3
from semantic_pipeline import load_core
from vlm_proposer import parse_proposals


def proposal(**changes):
    value = {"proposal_id": "crop-right", "semantic": "crop", "box_xyxy": [10, 0, 20, 20],
             "relation": "unspecified", "rationale": "Synthetic fixture; not real perception"}
    value.update(changes)
    return value


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="field-proposal-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)
        image_bytes = cv2.imencode(".png", self.image)[1].tobytes()
        (self.source / "frame.png").write_bytes(image_bytes)
        digest = hashlib.sha256(image_bytes).hexdigest()
        self.frame = {"frame_id": "f0", "video_name": "synthetic.mp4", "video_sha256": "b" * 64,
                      "frame_index": 5, "capture_time_s": 0.5, "image_path": "frame.png",
                      "artifact_sha256": {"frame.png": digest}}
        self.manifest = {"schema_version": "field_boundary_run_v1", "frames": [self.frame]}
        self.document = {"schema_version": "field_boundary_proposals_v1", "proposal_source": "human_review",
                         "frames": [{"frame_id": "f0", "image_sha256": digest, "proposals": [proposal()]}]}
        self.path = self.root / "proposals.json"
        self.save()

    def save(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def prepare(self):
        self.save()
        return prepare_inputs(self.source, self.path)

    def test_explicit_sources_and_image_identity_are_preserved(self):
        document, prepared, digest, manifest_digest = self.prepare()
        self.assertEqual(document["proposal_source"], "human_review")
        self.assertEqual(prepared[0]["queries"][0]["prompt"], "agricultural crops")
        self.assertEqual(prepared[0]["image_bytes"], (self.source / "frame.png").read_bytes())
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(manifest_digest), 64)
        self.assertEqual(sorted(path.name for path in self.source.iterdir()), ["frame.png", "manifest.json"])

    def test_wrong_image_or_manifest_hash_rejected(self):
        for target in (self.document["frames"][0], self.frame["artifact_sha256"]):
            key = "image_sha256" if "image_sha256" in target else "frame.png"
            old = target[key]
            target[key] = "a" * 64
            with self.assertRaisesRegex(ValueError, "hash"):
                self.prepare()
            target[key] = old
        (self.source / "frame.png").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.prepare()

    def test_no_model_or_output_on_preflight_failure(self):
        weights = self.root / "synthetic.pt"
        weights.write_bytes(b"not actual weights")
        args = argparse.Namespace(results=self.source, proposals=self.path, output=self.root / "out",
                                  weights=weights, clip_dir=None, confidence=0.3, device="cpu", threads=1)
        self.document["frames"][0]["image_sha256"] = "wrong"
        self.save()
        with patch("run_proposals.LocalSAM3") as model, self.assertRaises(ValueError):
            run(args)
        model.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_duplicate_unknown_and_unsafe_frame_ids(self):
        original = copy.deepcopy(self.document)
        for frames in ([original["frames"][0]] * 2,
                       [{**original["frames"][0], "frame_id": "missing"}],
                       [{**original["frames"][0], "frame_id": "../outside"}]):
            self.document["frames"] = frames
            with self.assertRaises(ValueError):
                self.prepare()

    def test_budget_and_source_validation(self):
        original = copy.deepcopy(self.document)
        for frames in ([], original["frames"] * 17, {}):
            self.document["frames"] = frames
            with self.assertRaises(ValueError):
                self.prepare()
        self.document = original
        for source in ("cloud", "auto_vlm", None, 123):
            self.document["proposal_source"] = source
            with self.assertRaises(ValueError):
                self.prepare()
        for source in ("assistant_visual_review", "human_review", "local_vlm", "cloud_vlm"):
            self.document["proposal_source"] = source
            self.prepare()

    def test_malformed_or_unbounded_proposals_rejected(self):
        invalid = [[], [proposal()] * 9, [proposal()] * 2,
                   [proposal(semantic="safe_headland")], [proposal(semantic="crop", relation="ego_candidate")],
                   [proposal(relation="verified_ego")], [proposal(proposal_id="../escape")],
                   [proposal(rationale="")], [proposal(rationale="x" * 501)],
                   [proposal(box_xyxy=[0, 0, 21, 20])], [proposal(box_xyxy=[0, 0, 0, 20])],
                   [proposal(box_xyxy=[0, 0, 20.0, 20])], [proposal(box_xyxy=[False, 0, 20, 20])],
                   [proposal(box_xyxy=[0, 0, 20])], [proposal(command="arbitrary")]]
        for proposals in invalid:
            with self.subTest(proposals=proposals):
                self.document["frames"][0]["proposals"] = proposals
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_extra_top_fields_and_large_document_rejected(self):
        self.document["model_command"] = "not allowed"
        with self.assertRaises(ValueError):
            self.prepare()
        self.path.write_bytes(b" " * 262145)
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            prepare_inputs(self.source, self.path)

    def test_source_path_escape_is_rejected(self):
        self.frame["image_path"] = "../outside.png"
        with self.assertRaises(ValueError):
            self.prepare()

    def analyze(self, changes=None, mask=None, score=0.9, source="human_review"):
        query = {**proposal(**(changes or {})), "prompt": "vehicle" if changes and changes.get("semantic") == "vehicle" else "agricultural crops"}
        if mask is None:
            mask = np.zeros((20, 20), dtype=bool)
            mask[:, 10:] = True
        detection = {"proposal_id": query["proposal_id"], "semantic": query["semantic"], "prompt": query["prompt"],
                     "box_xyxy": query["box_xyxy"], "score": score, "mask": mask, "bbox_xyxy": [10, 0, 20, 20]}
        return assess_predictions(self.image, [query], [detection], "f0", source, self.core)

    def test_local_vlm_warning_is_not_claimed_as_reviewed(self):
        reviewed = self.analyze()[0]["warnings"][0]
        vlm = self.analyze(source="local_vlm")[0]["warnings"][0]
        self.assertIn("not autonomous VLM perception", reviewed)
        self.assertIn("not verified ground truth", vlm)
        self.assertIn("Reviewed boxes", reviewed)

    def test_consistent_is_neither_semantic_truth_nor_metric_map(self):
        evidence, instances, arrays = self.analyze()
        self.assertEqual(instances[0]["spatial_check"]["status"], "spatially_consistent")
        self.assertFalse(instances[0]["semantic_verified"])
        self.assertEqual(evidence["unknown_fraction"], 1)
        self.assertIsNone(evidence["conflict_fraction"])
        self.assertEqual(evidence["boundaries"], [])
        self.assertIsNone(evidence["world_map"])
        self.assertNotIn("surface_labels", arrays)
        self.assertEqual(evidence["stage_prediction"], "UNKNOWN")
        self.assertFalse(evidence["hypothesis"]["maneuver_verified"])

    def test_high_score_outside_mask_retained_without_clipping(self):
        mask = np.ones((20, 20), dtype=bool)
        evidence, instances, arrays = self.analyze(mask=mask, score=0.999)
        self.assertEqual(instances[0]["spatial_check"]["status"], "outside_proposal")
        self.assertEqual(instances[0]["status"], "spatial_rejected")
        np.testing.assert_array_equal(arrays["candidate_0000"], mask)
        self.assertEqual(evidence["proposals"][0]["status"], "spatial_rejected")

    def test_clean_rendering_keeps_consistent_and_flagged_drops_noise(self):
        consistent = np.zeros((20, 20), dtype=bool)
        consistent[4:16, 12:18] = True  # Interior region so its contour is not covered by the cyan box.
        huge = np.ones((20, 20), dtype=bool)
        queries = [
            {**proposal(proposal_id="keep"), "prompt": "agricultural crops"},
            {**proposal(proposal_id="wide"), "semantic": "ground", "prompt": "bare soil"},
            {**proposal(proposal_id="junk"), "semantic": "ground", "prompt": "bare soil"},
        ]
        detections = []
        for query, mask, score in ((queries[0], consistent, 0.9), (queries[1], huge, 0.9), (queries[2], huge, 0.1)):
            detections.append({"proposal_id": query["proposal_id"], "semantic": query["semantic"], "prompt": query["prompt"],
                               "box_xyxy": query["box_xyxy"], "score": score, "mask": mask, "bbox_xyxy": [0, 0, 20, 20]})
        evidence, instances, _ = assess_predictions(self.image, queries, detections, "f0", "human_review", self.core)
        self.assertEqual([instance["spatial_check"]["status"] for instance in instances],
                         ["spatially_consistent", "outside_proposal", "outside_proposal"])
        self.assertEqual([instance["box_mask_disagreement"] for instance in instances], [False, True, False])
        self.assertEqual(len(evidence["instances"]), 3)
        full = render_candidates(self.image, instances, queries)
        clean = render_candidates(self.image, instances, queries, clean=True)
        self.assertTrue(np.any(np.all(full == (35, 35, 240), axis=-1)))
        self.assertFalse(np.any(np.all(clean == (35, 35, 240), axis=-1)))
        self.assertTrue(np.any(np.all(clean == (255, 255, 255), axis=-1)))
        self.assertTrue(np.any(np.all(clean == (0, 165, 255), axis=-1)))
        np.testing.assert_array_equal(clean, render_candidates(self.image, instances[:2], queries))

    def test_ego_candidate_is_not_removed_from_hazard_layer(self):
        evidence, instances, _ = self.analyze({"semantic": "vehicle", "relation": "ego_candidate"})
        self.assertEqual(len(evidence["hazards"]), 1)
        self.assertFalse(evidence["hazards"][0]["suppressed_as_ego"])
        self.assertFalse(instances[0]["ego_vehicle_verified"])
        self.assertEqual(instances[0]["relation_hypothesis"], "ego_candidate")

    def test_pole_is_proposable_and_routed_to_hazards(self):
        self.document["frames"][0]["proposals"] = [proposal(semantic="pole")]
        _, prepared, _, _ = self.prepare()
        self.assertEqual(prepared[0]["queries"][0]["prompt"], "utility pole")
        evidence, instances, _ = self.analyze({"semantic": "pole"})
        self.assertEqual(instances[0]["status"], "spatial_candidate_semantics_unverified")
        self.assertEqual(evidence["hazards"][0]["semantic"], "pole")
        self.assertEqual(evidence["regions"], [])

    def test_box_mask_disagreement_flags_only_plausible_rejections(self):
        _, instances, _ = self.analyze(mask=np.ones((20, 20), bool), score=0.9)
        self.assertEqual(instances[0]["status"], "spatial_rejected")
        self.assertTrue(instances[0]["box_mask_disagreement"])
        _, instances, _ = self.analyze(mask=np.ones((20, 20), bool), score=0.39)
        self.assertFalse(instances[0]["box_mask_disagreement"])
        sliver = np.zeros((20, 20), dtype=bool)
        sliver[0:2, 0:2] = True
        _, instances, _ = self.analyze(mask=sliver, score=0.9)
        self.assertEqual(instances[0]["spatial_check"]["status"], "outside_proposal")
        self.assertFalse(instances[0]["box_mask_disagreement"])
        _, instances, _ = self.analyze(mask=np.zeros((20, 20), bool), score=0.9)
        self.assertEqual(instances[0]["spatial_check"]["status"], "empty_mask")
        self.assertFalse(instances[0]["box_mask_disagreement"])
        _, instances, _ = self.analyze()
        self.assertEqual(instances[0]["status"], "spatial_candidate_semantics_unverified")
        self.assertFalse(instances[0]["box_mask_disagreement"])

    def test_empty_detection_does_not_establish_absence(self):
        query = {**proposal(), "prompt": "agricultural crops"}
        evidence, instances, arrays = assess_predictions(self.image, [query], [], "f0", "human_review", self.core)
        self.assertEqual(evidence["proposals"][0]["status"], "no_detection")
        self.assertEqual(evidence["unknown_fraction"], 1)
        self.assertEqual(instances, [])
        self.assertEqual(arrays, {})

    def test_invalid_mask_or_score_rejected(self):
        for mask in (np.ones((2, 2), bool), np.ones((20, 20), float)):
            with self.assertRaises(ValueError):
                self.analyze(mask=mask)
        for score in (float("nan"), True, 1.1, -0.1):
            with self.assertRaises(ValueError):
                self.analyze(score=score)

    def test_detection_cannot_change_proposal_identity(self):
        query = {**proposal(), "prompt": "agricultural crops"}
        detection = {"proposal_id": query["proposal_id"], "semantic": "crop", "prompt": query["prompt"],
                     "box_xyxy": query["box_xyxy"], "score": 0.8, "mask": np.ones((20, 20), bool), "bbox_xyxy": [0, 0, 20, 20]}
        for change in ({"proposal_id": "different"}, {"semantic": "ground"}, {"prompt": "field"}, {"box_xyxy": [0, 0, 20, 20]}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "bound proposal"):
                assess_predictions(self.image, [query], [{**detection, **change}], "f0", "human_review", self.core)

    def test_each_instance_has_one_shared_frame_evidence_group(self):
        query = {**proposal(), "prompt": "agricultural crops"}
        detection = {"proposal_id": query["proposal_id"], "semantic": "crop", "prompt": query["prompt"],
                     "box_xyxy": query["box_xyxy"], "score": 0.8, "mask": np.ones((20, 20), bool), "bbox_xyxy": [0, 0, 20, 20]}
        evidence, _, arrays = assess_predictions(self.image, [query], [detection, detection], "f0", "human_review", self.core)
        self.assertEqual(len(arrays), 2)
        self.assertEqual({instance["evidence_group"] for instance in evidence["instances"]}, {"f0"})

    def test_mocked_runner_preserves_source_and_rejects_existing_output(self):
        weights = self.root / "synthetic.pt"
        weights.write_bytes(b"not actual weights")
        args = argparse.Namespace(results=self.source, proposals=self.path, output=self.root / "out",
                                  weights=weights, clip_dir=None, confidence=0.3, device="cpu", threads=1)
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch("run_proposals.LocalSAM3") as model:
            model.return_value.metadata = {"backend": "SYNTHETIC_TEST_ONLY"}
            model.return_value.predict_regions.return_value = ([], {"total_seconds": 0})
            manifest = run(args)
            self.assertEqual(len(manifest["frames"]), 1)
            exported = manifest["frames"][0]
            self.assertEqual(len(exported["proposal_views"]), 1)
            self.assertEqual(exported["overlay_passed_path"], "f0/overlay_passed.jpg")
            self.assertEqual(exported["proposal_views"][0]["overlay_passed_path"], "f0/proposal_crop-right_passed.jpg")
            for relative in (exported["overlay_path"], exported["overlay_passed_path"],
                             exported["proposal_views"][0]["overlay_path"],
                             exported["proposal_views"][0]["overlay_passed_path"]):
                self.assertTrue((args.output / relative).is_file(), relative)
                self.assertIn(relative, exported["artifact_sha256"])
            store = ResultStore(args.output)
            self.assertEqual(store.manifest["frames"][0]["overlay_passed_path"], "f0/overlay_passed.jpg")
            self.assertEqual(store.read_file("f0/overlay_passed.jpg")[1], "image/jpeg")
            model.reset_mock()
            with self.assertRaisesRegex(ValueError, "new directory"):
                run(args)
            model.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        with np.load(args.output / "f0" / "masks.npz", allow_pickle=False) as masks:
            self.assertNotIn("surface_labels", masks)
        from fuse_metric import fuse
        with self.assertRaisesRegex(ValueError, "Unverified region proposals"):
            fuse(args.output, self.root / "not-read-geometry.json")


    def test_instance_cap_keeps_top_scores_and_records_audit(self):
        weights = self.root / "synthetic.pt"
        weights.write_bytes(b"not actual weights")

        def detection(score):
            return {"proposal_id": "crop-right", "semantic": "crop", "prompt": "agricultural crops",
                    "box_xyxy": [10, 0, 20, 20], "score": score,
                    "mask": np.ones((20, 20), bool), "bbox_xyxy": [10, 0, 20, 20]}

        detections = [detection(0.9), detection(0.5), detection(0.7)]
        args = argparse.Namespace(results=self.source, proposals=self.path, output=self.root / "capped",
                                  weights=weights, clip_dir=None, confidence=0.3, device="cpu", threads=1,
                                  max_instances_per_proposal=2)
        with patch("run_proposals.LocalSAM3") as model:
            model.return_value.metadata = {"backend": "SYNTHETIC_TEST_ONLY"}
            model.return_value.predict_regions.return_value = (detections, {"total_seconds": 0})
            run(args)
        evidence = json.loads((args.output / "f0" / "evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(len(evidence["instances"]), 2)
        self.assertEqual({round(i["score"], 6) for i in evidence["instances"]}, {0.9, 0.7})
        self.assertEqual(evidence["instance_cap"],
                         {"limit": 2, "dropped_per_proposal": {"crop-right": 1}, "dropped_total": 1,
                          "dropped_passed_total": 0})
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["configuration"]["max_instances_per_proposal"], 2)

    def test_instance_cap_protects_passed_instances_over_higher_scores(self):
        weights = self.root / "synthetic.pt"
        weights.write_bytes(b"not actual weights")

        def detection(score, inside_only):
            mask = np.zeros((20, 20), bool)
            mask[:, 10:] = True
            if not inside_only:
                mask[:, :10] = True
            bbox = [10, 0, 20, 20] if inside_only else [0, 0, 20, 20]
            return {"proposal_id": "crop-right", "semantic": "crop", "prompt": "agricultural crops",
                    "box_xyxy": [10, 0, 20, 20], "score": score, "mask": mask, "bbox_xyxy": bbox}

        detections = [detection(0.9, False), detection(0.3, True)]
        args = argparse.Namespace(results=self.source, proposals=self.path, output=self.root / "capped2",
                                  weights=weights, clip_dir=None, confidence=0.3, device="cpu", threads=1,
                                  max_instances_per_proposal=1)
        with patch("run_proposals.LocalSAM3") as model:
            model.return_value.metadata = {"backend": "SYNTHETIC_TEST_ONLY"}
            model.return_value.predict_regions.return_value = (detections, {"total_seconds": 0})
            run(args)
        evidence = json.loads((args.output / "f0" / "evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(len(evidence["instances"]), 1)
        instance = evidence["instances"][0]
        self.assertEqual(instance["spatial_check"]["status"], "spatially_consistent")
        self.assertEqual(round(instance["score"], 6), 0.3)
        self.assertEqual(evidence["instance_cap"],
                         {"limit": 1, "dropped_per_proposal": {"crop-right": 1}, "dropped_total": 1,
                          "dropped_passed_total": 0})


class CapVerifiedDetectionsTests(unittest.TestCase):
    @staticmethod
    def detection(proposal_id, score, marker):
        return {"proposal_id": proposal_id, "score": score, "marker": marker}

    def test_none_limit_returns_input_unchanged(self):
        detections = [self.detection("p0", 0.1, 0), self.detection("p1", 0.2, 1)]
        kept, audit = cap_verified_detections(detections, [True, False], None)
        self.assertIs(kept, detections)
        self.assertIsNone(audit)

    def test_passed_instances_outrank_higher_scoring_rejected(self):
        detections = [
            self.detection("p0", 0.9, 0), self.detection("p0", 0.3, 1),
            self.detection("p0", 0.8, 2), self.detection("p0", 0.5, 3),
        ]
        passed = [False, True, False, True]
        kept, audit = cap_verified_detections(detections, passed, 2)
        self.assertEqual([d["marker"] for d in kept], [1, 3])
        self.assertEqual(audit, {"limit": 2, "dropped_per_proposal": {"p0": 2},
                                 "dropped_total": 2, "dropped_passed_total": 0})

    def test_dropped_passed_counted_when_proposal_has_more_passed_than_limit(self):
        detections = [self.detection("p0", 0.9, 0), self.detection("p0", 0.8, 1),
                      self.detection("p0", 0.7, 2), self.detection("p0", 0.2, 3)]
        passed = [True, True, True, False]
        kept, audit = cap_verified_detections(detections, passed, 2)
        self.assertEqual([d["marker"] for d in kept], [0, 1])
        self.assertEqual(audit, {"limit": 2, "dropped_per_proposal": {"p0": 2},
                                 "dropped_total": 2, "dropped_passed_total": 1})

    def test_score_breaks_ties_within_status_and_original_order_breaks_score_ties(self):
        detections = [
            self.detection("p0", 0.5, 0), self.detection("p0", 0.9, 1),
            self.detection("p1", 0.7, 2), self.detection("p0", 0.6, 3),
            self.detection("p1", 0.4, 4),
        ]
        kept, audit = cap_verified_detections(detections, [False] * 5, 2)
        # all rejected: p0 keeps its two best (markers 1 and 3), p1 keeps both
        self.assertEqual([d["marker"] for d in kept], [1, 2, 3, 4])
        self.assertEqual(audit, {"limit": 2, "dropped_per_proposal": {"p0": 1},
                                 "dropped_total": 1, "dropped_passed_total": 0})
        ties = [self.detection("p", 0.5, i) for i in range(3)]
        kept, audit = cap_verified_detections(ties, [False] * 3, 2)
        self.assertEqual([d["marker"] for d in kept], [0, 1])

    def test_invalid_limit_rejected(self):
        with self.assertRaises(ValueError):
            cap_verified_detections([], [], 0)


class ArrayTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class RegionAdapterTests(unittest.TestCase):
    def test_one_encoder_and_one_box_per_query(self):
        model = LocalSAM3.__new__(LocalSAM3)
        image = np.zeros((20, 20, 3), np.uint8)
        queries = [{**proposal(proposal_id=f"p{i}"), "prompt": "agricultural crops"} for i in range(2)]
        with patch.object(model, "predictor", create=True) as predictor:
            predictor.inference_features.return_value = (
                ArrayTensor(np.ones((1, 20, 20), bool)), ArrayTensor(np.array([[0, 0, 20, 20, 0.8, 0]])))
            detections, timing = model.predict_regions(image, queries)
            self.assertEqual(predictor.set_image.call_count, 1)
            self.assertEqual(predictor.reset_image.call_count, 1)
            self.assertEqual(predictor.inference_features.call_count, 2)
            for call in predictor.inference_features.call_args_list:
                self.assertEqual(call.kwargs, {"text": ["agricultural crops"], "bboxes": [[10, 0, 20, 20]], "labels": [1]})
        self.assertEqual([item["proposal_id"] for item in detections], ["p0", "p1"])
        self.assertEqual(len(timing["prompts"]), 2)

    def test_region_budget_and_bad_boxes_fail_before_encoder(self):
        model = LocalSAM3.__new__(LocalSAM3)
        query = {**proposal(), "prompt": "agricultural crops"}
        for queries in ([], [query] * 9, [{**query, "box_xyxy": [-1, 0, 20, 20]}]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                model.predict_regions(np.zeros((20, 20, 3), np.uint8), queries)


class RerenderPassedVariantsTests(unittest.TestCase):
    def test_passed_variants_are_added_without_rewriting_existing_artifacts(self):
        temp = tempfile.TemporaryDirectory(prefix="rerender-passed-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        folder = root / "f0"
        folder.mkdir()
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        (folder / "frame.jpg").write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
        consistent = np.zeros((20, 20), dtype=bool)
        consistent[:, 10:] = True
        rejected = np.ones((20, 20), dtype=bool)
        np.savez_compressed(folder / "masks.npz", candidate_0000=consistent, candidate_0001=rejected)
        proposals = [
            {"proposal_id": "p1", "semantic": "crop", "box_xyxy": [0, 0, 20, 20],
             "relation": "unspecified", "rationale": "synthetic", "prompt": "agricultural crops"},
            {"proposal_id": "p2", "semantic": "ground", "box_xyxy": [0, 0, 20, 20],
             "relation": "unspecified", "rationale": "synthetic", "prompt": "bare soil"},
        ]
        instances = [
            {"proposal_id": "p1", "semantic": "crop", "mask_key": "candidate_0000", "bbox_xyxy": [10, 0, 20, 20],
             "score": 0.9, "status": "spatial_candidate_semantics_unverified",
             "spatial_check": {"status": "spatially_consistent", "outside_fraction": 0.0}},
            {"proposal_id": "p2", "semantic": "ground", "mask_key": "candidate_0001", "bbox_xyxy": [0, 0, 20, 20],
             "score": 0.9, "status": "spatial_rejected",
             "spatial_check": {"status": "outside_proposal", "outside_fraction": 0.5}},
        ]
        (folder / "evidence.json").write_text(
            json.dumps({"proposals": proposals, "instances": instances}), encoding="utf-8")
        item = {"frame_id": "f0", "image_path": "f0/frame.jpg", "overlay_path": "f0/overlay.jpg",
                "evidence_path": "f0/evidence.json", "masks_path": "f0/masks.npz",
                "proposal_views": [{"proposal_id": "p1", "label": "p1", "overlay_path": "f0/proposal_p1.jpg"},
                                   {"proposal_id": "p2", "label": "p2", "overlay_path": "f0/proposal_p2.jpg"}],
                "artifact_sha256": {"f0/overlay.jpg": "a" * 64, "f0/proposal_p1.jpg": "b" * 64,
                                    "f0/proposal_p2.jpg": "c" * 64}}
        for relative in ("f0/overlay.jpg", "f0/proposal_p1.jpg", "f0/proposal_p2.jpg"):
            (root / relative).write_bytes(b"sentinel never rewritten")
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        add_proposal_passed_variants(root, item)
        self.assertEqual(item["overlay_passed_path"], "f0/overlay_passed.jpg")
        self.assertEqual(item["proposal_views"][0]["overlay_passed_path"], "f0/proposal_p1_passed.jpg")
        self.assertEqual(item["proposal_views"][1]["overlay_passed_path"], "f0/proposal_p2_passed.jpg")
        for relative in ("f0/overlay_passed.jpg", "f0/proposal_p1_passed.jpg", "f0/proposal_p2_passed.jpg"):
            self.assertEqual(item["artifact_sha256"][relative], rerender_file_hash(root / relative))
        disk_image = cv2.imdecode(np.fromfile(root / "f0/frame.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
        expected = render_candidates(disk_image, [{**instances[0], "mask": consistent}], proposals, clean=True)
        self.assertEqual((root / "f0/overlay_passed.jpg").read_bytes(),
                         cv2.imencode(".jpg", expected)[1].tobytes())
        view_expected = render_candidates(disk_image, [], [proposals[1]], clean=True)
        self.assertEqual((root / "f0/proposal_p2_passed.jpg").read_bytes(),
                         cv2.imencode(".jpg", view_expected)[1].tobytes())
        after = {p.name: p.read_bytes() for p in folder.iterdir()}
        for name, data in before.items():
            self.assertEqual(after[name], data, name)


class VLMProposerTests(unittest.TestCase):
    def item(self, **changes):
        value = {"semantic": "crop", "box_xyxy": [10, 20, 300, 400], "relation": "unspecified",
                 "rationale": "left side green rows"}
        value.update(changes)
        return value

    def parse(self, items, text=None, width=100, height=80):
        body = text if text is not None else json.dumps({"proposals": items})
        return parse_proposals(body, width=width, height=height)

    def test_relative_0_1000_boxes_are_rescaled_to_pixels(self):
        accepted, rejected = self.parse([self.item(box_xyxy=[14.0, 25.0, 304.0, 412.4])])
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["box_xyxy"], [1, 2, 30, 33])
        self.assertEqual(accepted[0]["proposal_id"], "vlm-00")
        self.assertEqual(accepted[0]["model_box_raw"], [14.0, 25.0, 304.0, 412.4])

    def test_full_range_box_covers_whole_image(self):
        accepted, rejected = self.parse([self.item(box_xyxy=[0, 0, 1000, 1000])])
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["box_xyxy"], [0, 0, 100, 80])

    def test_recorded_qwen3_vl_vehicle_box_scales_as_expected(self):
        body = json.dumps({"proposals": [self.item(
            semantic="vehicle", relation="ego_candidate", box_xyxy=[205, 670, 999, 998])]})
        accepted, rejected = parse_proposals(body, width=1080, height=1280)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["box_xyxy"], [221, 858, 1079, 1277])

    def test_invalid_items_are_rejected_with_reasons_not_repaired(self):
        invalid = [self.item(semantic="safe_zone"), self.item(relation="ego_candidate"),
                   self.item(relation="verified"), self.item(box_xyxy=[0, 0, 1001, 999]),
                   self.item(box_xyxy=[-1, 0, 200, 200]), self.item(box_xyxy=[0, 0, 200]),
                   self.item(box_xyxy=["0", 0, 200, 200]), self.item(box_xyxy=[True, 0, 200, 200]),
                   self.item(box_xyxy=[float("nan"), 0, 200, 200]),
                   self.item(box_xyxy=[250, 0, 250, 999]),
                   self.item(rationale=""), self.item(rationale="x" * 501),
                   self.item(extra_field="rm -rf"), {"semantic": "crop"}]
        for item in invalid:
            with self.subTest(item=item):
                accepted, rejected = self.parse([item])
                self.assertEqual(accepted, [])
                self.assertTrue(rejected[0]["reason"])
                self.assertIn("raw", rejected[0])
        _, rejected = self.parse([self.item(box_xyxy=[0, 0, 1001, 999])])
        self.assertEqual(rejected[0]["reason"], "box_coordinate_invalid")
        _, rejected = self.parse([self.item(box_xyxy=[250, 0, 250, 999])])
        self.assertEqual(rejected[0]["reason"], "box_empty_after_scaling")

    def test_non_json_and_missing_array_are_rejected(self):
        for text in ("no json here", "{'proposals': []}", '{"proposals": 7}', '{"other": []}',
                     'prefix {"proposals": []} suffix'):
            with self.subTest(text=text):
                accepted, rejected = self.parse(None, text=text)
                if text.startswith("prefix"):
                    self.assertEqual(accepted, [])
                    self.assertEqual(rejected, [])
                else:
                    self.assertEqual(accepted, [])
                    self.assertTrue(rejected)

    def test_budget_over_eight_is_truncated_and_recorded(self):
        items = [self.item(box_xyxy=[10 * i, 20, 300 + 10 * i, 400]) for i in range(10)]
        accepted, rejected = self.parse(items)
        self.assertEqual([item["proposal_id"] for item in accepted], [f"vlm-{i:02d}" for i in range(8)])
        self.assertEqual({item["reason"] for item in rejected}, {"budget_exceeded"})

    def test_exact_duplicate_boxes_are_rejected_with_provenance(self):
        first = self.item(semantic="ground", box_xyxy=[0, 368, 1000, 538])
        second = self.item(semantic="harvested_ground", box_xyxy=[0, 368, 1000, 538])
        accepted, rejected = self.parse([first, second])
        self.assertEqual([p["semantic"] for p in accepted], ["ground"])
        self.assertEqual(accepted[0]["proposal_id"], "vlm-00")
        self.assertEqual(rejected[0]["reason"], "duplicate_box_of:vlm-00")
        self.assertEqual(rejected[0]["raw"], second)

    def test_same_semantic_same_box_is_also_a_duplicate(self):
        accepted, rejected = self.parse([self.item(), self.item()])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["reason"], "duplicate_box_of:vlm-00")

    def test_near_overlap_and_duplicates_after_scaling_are_kept(self):
        near = [self.item(box_xyxy=[0, 368, 999, 538]), self.item(box_xyxy=[1, 368, 1000, 538])]
        accepted, rejected = self.parse(near, width=1000, height=1000)
        self.assertEqual([p["box_xyxy"] for p in accepted], [[0, 368, 999, 538], [1, 368, 1000, 538]])
        self.assertEqual(rejected, [])
        raw_equal_after_scaling = [self.item(box_xyxy=[0, 368, 1000, 538]),
                                   self.item(box_xyxy=[0.4, 368, 999.6, 538])]
        accepted, rejected = self.parse(raw_equal_after_scaling, width=1000, height=1000)
        self.assertEqual([p["box_xyxy"] for p in accepted], [[0, 368, 1000, 538]])
        self.assertEqual(rejected[0]["reason"], "duplicate_box_of:vlm-00")

    def test_stream_of_identical_items_yields_one_accepted(self):
        accepted, rejected = self.parse([self.item() for _ in range(10)])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 9)
        self.assertTrue(all(r["reason"].startswith("duplicate_box_of:") for r in rejected))


if __name__ == "__main__":
    unittest.main()
