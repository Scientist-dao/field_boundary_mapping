import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_temporal_consistency import (
    build_chains, build_json_payload, chain_metrics, chain_units, iou,
    semantic_run_stats, units_from_groups,
)


def make_instance(instance_id, semantic, box, status="spatially_consistent"):
    return {"instance_id": instance_id, "semantic": semantic, "bbox": box, "score": 0.9, "status": status}


def make_frame(frame_id, second, instances):
    return {
        "frame_id": frame_id, "second": second, "capture_time_s": second,
        "proposal_source": "cloud_vlm", "proposals": [], "instances": instances,
    }


class IouTests(unittest.TestCase):
    def test_identical_box_is_one_and_disjoint_is_zero(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 0, 30, 10]), 0.0)


class ChainTests(unittest.TestCase):
    def test_stable_instance_yields_single_full_length_chain(self):
        box = [10, 10, 50, 50]
        frames = [make_frame(f"f{i}", i, [make_instance(f"i{i}", "vehicle", box)]) for i in (0, 2, 4)]
        chains, linkage = build_chains(frames, 0.2)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["steps"][0]["second"], 0)
        self.assertEqual(len(chains[0]["steps"]), 3)
        ids = {linkage[(f"f{s}", f"i{s}")] for s in (0, 2, 4)}
        self.assertEqual(ids, {chains[0]["chain_id"]})

    def test_gap_breaks_chain_because_linking_is_consecutive_only(self):
        box = [0, 0, 100, 100]
        frames = [
            make_frame("f0", 0, [make_instance("a", "crop", box)]),
            make_frame("f1", 2, [make_instance("b", "crop", box)]),
            make_frame("f2", 4, []),
            make_frame("f3", 6, [make_instance("c", "crop", box)]),
            make_frame("f4", 8, [make_instance("d", "crop", box)]),
        ]
        chains, _ = build_chains(frames, 0.2)
        self.assertEqual(sorted(len(c["steps"]) for c in chains), [2, 2])
        self.assertNotEqual(chains[0]["chain_id"], chains[1]["chain_id"])

    def test_rejected_instances_never_join_chains(self):
        frames = [
            make_frame("f0", 0, [make_instance("a", "ditch", [0, 0, 10, 10], status="outside_proposal")]),
            make_frame("f1", 2, [make_instance("b", "ditch", [0, 0, 10, 10], status="spatially_consistent")]),
        ]
        chains, linkage = build_chains(frames, 0.2)
        self.assertEqual(len(chains), 1)
        self.assertEqual(len(chains[0]["steps"]), 1)
        self.assertNotIn(("f0", "a"), linkage)

    def test_semantic_flip_count_and_modal_stats(self):
        box = [0, 0, 100, 100]
        frames = [
            make_frame("f0", 0, [make_instance("a", "vehicle", box)]),
            make_frame("f1", 2, [make_instance("b", "vehicle", box)]),
            make_frame("f2", 4, [make_instance("c", "crop", box)]),
        ]
        chains, _ = build_chains(frames, 0.2)
        metrics = chain_metrics(chains[0], [0, 2, 4, 6, 8])
        self.assertEqual(metrics["flip_count"], 1)
        self.assertEqual(metrics["modal_semantic"], "vehicle")
        self.assertEqual(metrics["modal_fraction"], round(2 / 3, 4))
        self.assertEqual(metrics["appearance_rate"], round(3 / 5, 4))
        self.assertEqual(metrics["semantic_sequence"], ["vehicle", "vehicle", "crop"])

    def test_overlapping_competitors_split_into_chain_and_singleton(self):
        frames = [
            make_frame("f0", 0, [make_instance("a", "vehicle", [0, 0, 100, 100])]),
            make_frame("f1", 2, [make_instance("x", "vehicle", [0, 0, 100, 100]),
                                 make_instance("y", "vehicle", [10, 10, 110, 110])]),
        ]
        chains, linkage = build_chains(frames, 0.2)
        lengths = sorted(len(c["steps"]) for c in chains)
        self.assertEqual(lengths, [1, 2])
        chain_with_two = next(c for c in chains if len(c["steps"]) == 2)
        self.assertEqual(chain_with_two["steps"][1]["instance_id"], "x")
        singleton = next(c for c in chains if len(c["steps"]) == 1)
        self.assertEqual(linkage[("f1", "y")], singleton["chain_id"])


class ChainUnitsTests(unittest.TestCase):
    def test_default_units_are_the_frames_passed_instances(self):
        frame = make_frame("f0", 0, [
            make_instance("a", "crop", [0, 0, 10, 10]),
            make_instance("b", "crop", [0, 0, 10, 10], status="spatial_rejected"),
        ])
        self.assertEqual([u["instance_id"] for u in chain_units(frame)], ["a"])

    def test_attached_group_units_override_the_instance_basis(self):
        frame = make_frame("f0", 0, [make_instance("a", "tarp", [0, 0, 10, 10])])
        frame["chain_units"] = [{"instance_id": "rep", "semantic": "tarp",
                                 "bbox": [0, 0, 10, 10], "score": 0.9}]
        self.assertEqual([u["instance_id"] for u in chain_units(frame)], ["rep"])


class UnitsFromGroupsTests(unittest.TestCase):
    @staticmethod
    def group(group_id, semantic, members, representative, has_passed, passed_count):
        return {"group_id": group_id, "semantic": semantic, "members": members,
                "member_count": len(members), "representative": representative,
                "has_passed": has_passed, "passed_count": passed_count}

    def test_one_unit_per_group_with_passed_and_none_for_all_rejected_groups(self):
        frame = make_frame("f0", 22, [
            make_instance("i1", "tarp", [0, 0, 100, 100]),
            make_instance("i2", "tarp", [10, 10, 90, 90]),
            make_instance("i3", "woody", [200, 0, 300, 50], status="spatial_rejected"),
        ])
        grouped = {"groups": [
            self.group("g-000", "tarp", ["i1", "i2"], "i2", True, 2),
            self.group("g-001", "woody", ["i3"], "i3", False, 0),
        ]}
        units = units_from_groups(frame, grouped)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["instance_id"], "i2")
        self.assertEqual(units[0]["semantic"], "tarp")
        self.assertEqual(units[0]["bbox"], [10, 10, 90, 90])
        self.assertEqual(units[0]["group_id"], "g-000")
        self.assertEqual(units[0]["member_count"], 2)
        self.assertEqual(units[0]["passed_count"], 2)


class GroupChainTests(unittest.TestCase):
    def test_parallel_nested_instance_chains_collapse_into_one_object_chain(self):
        box = [0, 0, 100, 100]
        raw_frames, group_frames = [], []
        for second in (22, 24, 26):
            instances = [make_instance(f"s{second}i{k}", "tarp", box) for k in range(3)]
            raw_frames.append(make_frame(f"f{second}", second, instances))
            frame = make_frame(f"f{second}", second, instances)
            frame["chain_units"] = [{
                "instance_id": f"s{second}i0", "semantic": "tarp", "bbox": box, "score": 0.95,
                "group_id": "g-000", "member_count": 3, "passed_count": 3,
            }]
            group_frames.append(frame)
        self.assertEqual(len(build_chains(raw_frames, 0.2)[0]), 3)
        chains, linkage = build_chains(group_frames, 0.2)
        self.assertEqual(len(chains), 1)
        self.assertEqual(len(chains[0]["steps"]), 3)
        self.assertEqual([s["second"] for s in chains[0]["steps"]], [22, 24, 26])
        self.assertEqual(linkage[("f22", "s22i0")], chains[0]["chain_id"])

    def test_group_payload_declares_basis_and_stays_review_only(self):
        box = [0, 0, 100, 100]
        frames = []
        for second in (22, 24):
            frame = make_frame(f"f{second}", second, [make_instance(f"i{second}", "tarp", box)])
            frame["chain_units"] = [{
                "instance_id": f"i{second}", "semantic": "tarp", "bbox": box, "score": 0.95,
                "group_id": "g-000", "member_count": 2, "passed_count": 2,
            }]
            frames.append(frame)
        chains, linkage = build_chains(frames, 0.2)
        payload = build_json_payload(Path("fake_run"), frames, chains, linkage, 0.2,
                                     basis="group_representatives",
                                     grouping={"duplicate_iou": 0.95})
        self.assertEqual(payload["schema_version"], "temporal_consistency_v2")
        self.assertEqual(payload["chain_basis"], "group_representatives")
        self.assertEqual(payload["grouping"]["duplicate_iou"], 0.95)
        self.assertFalse(payload["semantic_verified"])
        self.assertFalse(payload["annotation_used_for_inference"])
        step = payload["chains"][0]["steps"][0]
        self.assertEqual(step["group_id"], "g-000")
        self.assertEqual(step["member_count"], 2)
        unit = payload["per_frame"][0]["units"][0]
        self.assertEqual(unit["group_id"], "g-000")
        self.assertEqual(unit["chain_length"], 2)


class SemanticRunTests(unittest.TestCase):
    def test_intermittent_semantic_never_sustains_a_run(self):
        frames = [
            make_frame("f0", 0, [make_instance("a", "ditch", [0, 0, 10, 10])]),
            make_frame("f1", 2, [make_instance("b", "crop", [0, 0, 10, 10])]),
            make_frame("f2", 4, [make_instance("c", "ditch", [0, 0, 10, 10])]),
            make_frame("f3", 6, [make_instance("d", "crop", [0, 0, 10, 10])]),
            make_frame("f4", 8, [make_instance("e", "crop", [0, 0, 10, 10])]),
        ]
        summary = semantic_run_stats(frames)
        self.assertEqual(summary["ditch"]["longest_consecutive_run"], 1)
        self.assertEqual(summary["ditch"]["present_seconds"], [0, 4])
        self.assertEqual(summary["crop"]["longest_consecutive_run"], 2)

    def test_only_passed_instances_count(self):
        frames = [
            make_frame("f0", 0, [make_instance("a", "tarp", [0, 0, 10, 10], status="outside_proposal")]),
            make_frame("f1", 2, [make_instance("b", "tarp", [0, 0, 10, 10])]),
        ]
        self.assertEqual(semantic_run_stats(frames)["tarp"]["longest_consecutive_run"], 1)


class JsonPayloadTests(unittest.TestCase):
    def test_payload_is_evidence_only_never_verified_and_never_inferred_from_annotations(self):
        box = [0, 0, 100, 100]
        frames = [make_frame(f"f{i}", 2 * i, [make_instance(f"i{i}", "crop", box)]) for i in range(3)]
        chains, linkage = build_chains(frames, 0.2)
        payload = build_json_payload(Path("fake_run"), frames, chains, linkage, 0.2)
        self.assertFalse(payload["semantic_verified"])
        self.assertFalse(payload["annotation_used_for_inference"])
        self.assertEqual(payload["schema_version"], "temporal_consistency_v2")
        self.assertEqual(payload["chain_basis"], "instances")
        self.assertEqual(payload["run"], "fake_run")
        self.assertEqual(len(payload["chains"]), 1)
        first = payload["per_frame"][0]["units"][0]
        self.assertEqual(first["chain_length"], 3)
        self.assertEqual(first["appearance_rate"], 1.0)
        self.assertEqual(payload["semantic_summary"]["crop"]["longest_consecutive_run"], 3)
        self.assertEqual(payload["semantic_summary"]["crop"]["present_seconds"], [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
