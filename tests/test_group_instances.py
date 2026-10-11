import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_boundary_mapping.evidence.group_instances import (
    aggregate_frames, build_json_payload, group_frame, same_object, UnionFind,
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
            "score": item.get("score", 0.9), "bbox_xyxy": list(item["box"]),
            "spatial_check": {"status": item["status"]},
            "box_mask_disagreement": False,
        })
    evidence = {
        "frame_id": frame_id, "proposal_source": "cloud_vlm",
        "source": {"capture_time_s": second},
        "proposals": [],
        "instances": records,
    }
    folder = Path(root) / frame_id
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "masks.npz", **arrays)
    (folder / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return folder


def instance(proposal_id, semantic, box, status="spatially_consistent", score=0.9):
    return {"proposal_id": proposal_id, "semantic": semantic, "box": box,
            "status": status, "score": score}


class UnionFindTests(unittest.TestCase):
    def test_transitive_union_merges_three(self):
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))

    def test_disjoint_stay_separate(self):
        uf = UnionFind(["a", "b"])
        self.assertNotEqual(uf.find("a"), uf.find("b"))


class SameObjectTests(unittest.TestCase):
    def test_near_duplicate_threshold(self):
        self.assertTrue(same_object({"iou": 0.96, "contained": 0.96, "small": 10, "large": 11},
                                    0.95, 0.95, 0.8))
        self.assertFalse(same_object({"iou": 0.9, "contained": 0.9, "small": 10, "large": 11},
                                     0.95, 0.95, 0.8))

    def test_containment_rule_requires_area_gap(self):
        twin = {"iou": 0.6, "contained": 1.0, "small": 5, "large": 8}
        self.assertTrue(same_object(twin, 0.95, 0.95, 0.8))
        close_sizes = {"iou": 0.9, "contained": 0.99, "small": 10, "large": 11}
        self.assertFalse(same_object(close_sizes, 0.95, 0.95, 0.8))


class GroupFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="instance-groups-test-")
        self.addCleanup(self.temp.cleanup)

    def group(self, frame_id, second, instances):
        return group_frame(write_frame(self.temp.name, frame_id, second, instances))

    def test_transitive_duplicates_form_one_group(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "tarp", [2, 2, 8, 8]),
            instance("vlm-01", "tarp", [2, 2, 8, 8]),
            instance("vlm-02", "tarp", [2, 2, 8, 8]),
        ])
        self.assertEqual(report["groups_total"], 1)
        group = report["groups"][0]
        self.assertEqual(group["member_count"], 3)
        self.assertEqual(group["proposals"], ["vlm-00", "vlm-01", "vlm-02"])
        self.assertEqual(report["groups_with_passed"], 1)

    def test_whole_part_twins_merge_into_one_object_group(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "tarp", [1, 1, 10, 10]),
            instance("vlm-01", "tarp", [4, 4, 6, 6]),
        ])
        self.assertEqual(report["groups_total"], 1)
        self.assertEqual(report["groups"][0]["member_count"], 2)

    def test_cross_semantic_overlap_never_merges(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "crop", [2, 2, 8, 8]),
            instance("vlm-01", "ground", [2, 2, 8, 8]),
        ])
        self.assertEqual(report["groups_total"], 2)
        self.assertEqual(len(report["conflict_pairs"]), 1)
        pair = report["conflict_pairs"][0]
        self.assertEqual(pair["semantic_pair"], "crop|ground")
        self.assertNotEqual(pair["group_a"], pair["group_b"])

    def test_cross_semantic_containment_keeps_both_groups(self):
        # a tiny tarp fragment inside a huge passed ground mask must stay its own group
        report = self.group("f0", 0, [
            instance("vlm-00", "ground", [1, 1, 15, 15]),
            instance("vlm-01", "tarp", [7, 7, 8, 8]),
        ])
        self.assertEqual(report["groups_total"], 2)
        semantics = {g["semantic"] for g in report["groups"]}
        self.assertEqual(semantics, {"ground", "tarp"})
        self.assertEqual(report["conflict_pairs"], [])

    def test_below_threshold_overlap_stays_two_objects(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "crop", [2, 2, 8, 8]),
            instance("vlm-01", "crop", [4, 4, 10, 10]),
        ])
        self.assertEqual(report["groups_total"], 2)

    def test_disjoint_instances_are_singleton_groups(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "crop", [1, 1, 4, 4]),
            instance("vlm-01", "ground", [8, 8, 12, 12]),
        ])
        self.assertEqual(report["groups_total"], 2)
        self.assertEqual(report["singleton_groups"], 2)
        self.assertEqual(report["groups_with_passed"], 2)

    def test_rejected_only_groups_counted_separately(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "woody_vegetation", [2, 2, 8, 8]),
            instance("vlm-01", "woody_vegetation", [2, 2, 8, 8], status="outside_proposal"),
            instance("vlm-02", "crop", [1, 1, 4, 4], status="outside_proposal"),
        ])
        self.assertEqual(report["groups_total"], 2)
        self.assertEqual(report["groups_with_passed"], 1)
        self.assertEqual(report["groups_all_rejected"], 1)

    def test_representative_prefers_passed_then_highest_score(self):
        report = self.group("f0", 0, [
            instance("vlm-00", "vehicle", [2, 2, 8, 8], status="outside_proposal", score=0.99),
            instance("vlm-01", "vehicle", [2, 2, 8, 8], score=0.4),
        ])
        group = report["groups"][0]
        self.assertEqual(group["representative"], "candidate_0001")
        self.assertTrue(group["representative_passed"])
        self.assertEqual(group["representative_score"], 0.4)


class AggregateAndPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="instance-groups-test-")
        self.addCleanup(self.temp.cleanup)
        self.thresholds = {"duplicate_iou": 0.95, "containment_fraction": 0.95,
                           "containment_area_ratio": 0.8, "conflict_iou": 0.5}

    def test_aggregate_per_semantic_counts(self):
        frames = [
            group_frame(write_frame(self.temp.name, "f0", 0, [
                instance("vlm-00", "tarp", [1, 1, 10, 10]),
                instance("vlm-01", "tarp", [4, 4, 6, 6]),
                instance("vlm-02", "crop", [1, 12, 4, 15]),
            ])),
            group_frame(write_frame(self.temp.name, "f2", 2, [
                instance("vlm-00", "tarp", [1, 1, 10, 10]),
            ])),
        ]
        totals = aggregate_frames(frames)
        self.assertEqual(totals["frames"], 2)
        self.assertEqual(totals["instances"], 4)
        self.assertEqual(totals["groups_total"], 3)
        self.assertEqual(totals["by_semantic"]["tarp"]["instances"], 3)
        self.assertEqual(totals["by_semantic"]["tarp"]["groups"], 2)
        self.assertEqual(totals["by_semantic"]["tarp"]["groups_with_passed"], 2)

    def test_payload_is_evidence_only_and_serializable(self):
        folder = write_frame(self.temp.name, "f0", 0, [instance("vlm-00", "crop", [1, 1, 4, 4])])
        frames = [group_frame(folder)]
        totals = aggregate_frames(frames)
        payload = build_json_payload(Path(self.temp.name), frames, totals, self.thresholds)
        self.assertEqual(payload["schema_version"], "instance_groups_v1")
        self.assertFalse(payload["semantic_verified"])
        self.assertFalse(payload["annotation_used_for_inference"])
        self.assertEqual(payload["proposal_source"], "cloud_vlm")
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
