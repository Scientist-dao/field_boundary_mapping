import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_instance_conflicts import (
    aggregate_reports, audit_frame, build_json_payload, pair_overlap,
)


def write_frame(root, frame_id, second, instances):
    arrays = {}
    records = []
    for index, item in enumerate(instances):
        key = f"candidate_{index:04d}"
        mask = np.zeros((16, 16), dtype=bool)
        x0, y0, x1, y1 = item["box"]
        mask[y0:y1, x0:x1] = True
        arrays[key] = mask
        records.append({
            "instance_id": key, "mask_key": key,
            "proposal_id": item["proposal_id"], "semantic": item["semantic"],
            "score": 0.9, "bbox_xyxy": list(item["box"]),
            "spatial_check": {"status": item["status"]},
            "box_mask_disagreement": item.get("disagreement", False),
        })
    evidence = {
        "frame_id": frame_id, "proposal_source": "cloud_vlm",
        "source": {"capture_time_s": second},
        "proposals": [{"proposal_id": p} for p in sorted({i["proposal_id"] for i in instances})],
        "instances": records,
    }
    folder = Path(root) / frame_id
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "masks.npz", **arrays)
    (folder / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return folder


class PairOverlapTests(unittest.TestCase):
    def test_pair_overlap_identical_and_disjoint(self):
        a = np.zeros((8, 8), dtype=bool)
        b = np.zeros((8, 8), dtype=bool)
        a[2:6, 2:6] = True
        b[2:6, 2:6] = True
        self.assertEqual(pair_overlap(a, b)["iou"], 1.0)
        self.assertEqual(pair_overlap(a, b)["contained"], 1.0)
        c = np.zeros((8, 8), dtype=bool)
        c[0:2, 0:2] = True
        self.assertEqual(pair_overlap(a, c)["iou"], 0.0)

    def test_pair_overlap_reports_smaller_mask(self):
        a = np.zeros((8, 8), dtype=bool)
        b = np.zeros((8, 8), dtype=bool)
        a[1:7, 1:7] = True  # area 36
        b[2:4, 2:4] = True  # area 4, fully inside a
        overlap = pair_overlap(a, b)
        self.assertFalse(overlap["small_is_a"])
        self.assertEqual(overlap["small"], 4)
        self.assertEqual(overlap["large"], 36)
        self.assertAlmostEqual(overlap["contained"], 1.0)
        self.assertAlmostEqual(overlap["iou"], 4 / 36)


class AuditFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="instance-conflicts-test-")
        self.addCleanup(self.temp.cleanup)

    def audit(self, frame_id, second, instances):
        return audit_frame(write_frame(self.temp.name, frame_id, second, instances))

    def test_identical_masks_across_proposals_are_cross_proposal_duplicates(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "ground", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
        ])
        self.assertEqual(len(report["duplicate_pairs"]), 1)
        pair = report["duplicate_pairs"][0]
        self.assertTrue(pair["cross_proposal"])
        self.assertFalse(pair["same_semantic"])
        self.assertEqual(pair["status"], "both_passed")
        self.assertEqual(report["containment_pairs"], [])

    def test_same_semantic_passed_overlap_is_duplicate_not_conflict(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "vehicle", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "vehicle", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
        ])
        self.assertEqual(len(report["duplicate_pairs"]), 1)
        self.assertEqual(report["conflicts"], [])

    def test_conflict_requires_both_passed_and_different_semantics(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "ground", "box": [2, 2, 8, 8],
             "status": "outside_proposal"},
        ])
        self.assertEqual(report["conflicts"], [])
        self.assertEqual(report["duplicate_pairs"][0]["status"], "one_passed")

    def test_rejected_twin_of_passed_instance_is_flagged_redundant(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "outside_proposal"},
        ])
        self.assertEqual(len(report["redundant_rejected"]), 1)
        entry = report["redundant_rejected"][0]
        self.assertEqual(entry["instance_id"], "candidate_0001")
        self.assertEqual(entry["matched_passed"], "candidate_0000")

    def test_rejected_without_passed_twin_is_not_redundant(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "outside_proposal"},
            {"proposal_id": "vlm-01", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "outside_proposal"},
        ])
        self.assertEqual(report["redundant_rejected"], [])
        self.assertEqual(report["duplicate_pairs"][0]["status"], "both_rejected")

    def test_whole_part_containment_detected(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "tarp", "box": [1, 1, 10, 10],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "tarp", "box": [4, 4, 6, 6],
             "status": "spatially_consistent"},
        ])
        self.assertEqual(len(report["containment_pairs"]), 1)
        pair = report["containment_pairs"][0]
        self.assertEqual(pair["small"], "candidate_0001")
        self.assertEqual(pair["large"], "candidate_0000")
        self.assertEqual(pair["contained_fraction"], 1.0)
        self.assertEqual(report["duplicate_pairs"], [])
        self.assertEqual(report["conflicts"], [])

    def test_disjoint_instances_produce_no_pairs(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [1, 1, 4, 4],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "ground", "box": [8, 8, 12, 12],
             "status": "spatially_consistent"},
        ])
        self.assertEqual(report["duplicate_pairs"], [])
        self.assertEqual(report["conflicts"], [])
        self.assertEqual(report["containment_pairs"], [])
        self.assertEqual(report["redundant_rejected"], [])

    def test_census_counts(self):
        report = self.audit("f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [1, 1, 5, 5],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [8, 1, 12, 5],
             "status": "spatially_consistent", "disagreement": True},
            {"proposal_id": "vlm-01", "semantic": "ground", "box": [1, 8, 12, 12],
             "status": "outside_proposal"},
        ])
        self.assertEqual(report["instances"], 3)
        self.assertEqual(report["passed"], 2)
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["flagged_disagreement"], 1)
        self.assertEqual(report["proposals"], 2)
        self.assertEqual(report["max_instances_per_proposal"], 2)
        self.assertEqual(report["second"], 0)


class AggregateAndPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="instance-conflicts-test-")
        self.addCleanup(self.temp.cleanup)
        self.thresholds = {"duplicate_iou": 0.95, "conflict_iou": 0.5,
                           "containment_fraction": 0.95, "containment_area_ratio": 0.8}

    def test_aggregate_totals_and_semantic_pairs(self):
        frame_a = write_frame(self.temp.name, "f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "ground", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
        ])
        frame_b = write_frame(self.temp.name, "f2", 2, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
            {"proposal_id": "vlm-01", "semantic": "ground", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
        ])
        reports = [audit_frame(frame_a), audit_frame(frame_b)]
        totals = aggregate_reports(reports)
        self.assertEqual(totals["frames"], 2)
        self.assertEqual(totals["instances"], 4)
        self.assertEqual(totals["passed"], 4)
        self.assertEqual(totals["duplicate_counts"]["total"], 2)
        self.assertEqual(totals["conflicts_total"], 2)
        self.assertEqual(totals["conflicts_by_semantic_pair"], {"crop|ground": 2})

    def test_payload_is_evidence_only(self):
        frame = write_frame(self.temp.name, "f0", 0, [
            {"proposal_id": "vlm-00", "semantic": "crop", "box": [2, 2, 8, 8],
             "status": "spatially_consistent"},
        ])
        reports = [audit_frame(frame)]
        totals = aggregate_reports(reports)
        payload = build_json_payload(Path("run"), reports, totals, self.thresholds)
        self.assertEqual(payload["schema_version"], "instance_conflicts_audit_v1")
        self.assertFalse(payload["semantic_verified"])
        self.assertFalse(payload["annotation_used_for_inference"])
        self.assertEqual(payload["frames"][0]["instances"], 1)
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
