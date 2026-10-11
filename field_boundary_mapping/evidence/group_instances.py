"""Group near-duplicate segmentation instances into derived object groups.

Read-only layer over a run produced by run_proposals.py. Per frame, instances
of the SAME semantic are merged into one object group when they are near
duplicates (mask IoU >= duplicate_iou) or whole/part twins (the smaller mask
is >= containment_fraction inside the larger while its area is at most
containment_area_ratio of the larger). Groups are connected components
(union-find) of that relation, so pair counts are never mistaken for object
counts. Cross-semantic overlaps are NEVER merged into one group: deciding
which name is right is a human review task; such overlaps are reported as
conflict pairs between groups instead.

This is a derived counting/review layer only: evidence.json, masks.npz and
the manifests stay untouched, no instance is deleted or edited, group counts
are review statistics (not verified object counts), and semantic_verified
remains false everywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from field_boundary_mapping.evidence.analyze_instance_conflicts import load_frame
from field_boundary_mapping.proposals.run_offline import write_json

BASE = Path(__file__).resolve().parents[2]   # 仓库根

DUPLICATE_IOU = 0.95
CONTAINMENT_FRACTION = 0.95
CONTAINMENT_AREA_RATIO = 0.8
CONFLICT_IOU = 0.5
PASSED = "spatially_consistent"


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def pair_stats(mask_a, mask_b):
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0 or area_a == 0 or area_b == 0:
        return {"iou": 0.0, "contained": 0.0, "small": 0, "large": 0}
    iou = inter / (area_a + area_b - inter)
    small = min(area_a, area_b)
    large = max(area_a, area_b)
    return {"iou": iou, "contained": inter / small, "small": small, "large": large}


def same_object(overlap, duplicate_iou, containment_fraction, containment_area_ratio):
    if overlap["iou"] >= duplicate_iou:
        return True
    return (overlap["contained"] >= containment_fraction
            and overlap["small"] <= containment_area_ratio * overlap["large"])


def group_frame(frame_dir, duplicate_iou=DUPLICATE_IOU,
                containment_fraction=CONTAINMENT_FRACTION,
                containment_area_ratio=CONTAINMENT_AREA_RATIO,
                conflict_iou=CONFLICT_IOU):
    evidence, instances = load_frame(frame_dir)
    uf = UnionFind([instance["instance_id"] for instance in instances])
    conflicts = []
    for i in range(len(instances)):
        a = instances[i]
        for j in range(i + 1, len(instances)):
            b = instances[j]
            if a["semantic"] != b["semantic"]:
                if (a["passed"] and b["passed"]
                        and pair_stats(a["mask"], b["mask"])["iou"] >= conflict_iou):
                    conflicts.append((a["instance_id"], b["instance_id"],
                                      "|".join(sorted((a["semantic"], b["semantic"])))))
                continue
            overlap = pair_stats(a["mask"], b["mask"])
            if same_object(overlap, duplicate_iou, containment_fraction, containment_area_ratio):
                uf.union(a["instance_id"], b["instance_id"])

    groups = {}
    for instance in instances:
        groups.setdefault(uf.find(instance["instance_id"]), []).append(instance)
    ordered = []
    for root, members in sorted(groups.items()):
        members = sorted(members, key=lambda m: m["instance_id"])
        passed_members = [m for m in members if m["passed"]]
        pool = passed_members if passed_members else members
        representative = sorted(pool, key=lambda m: (-m["score"], m["instance_id"]))[0]
        ordered.append({
            "group_id": f"g-{len(ordered):03d}",
            "semantic": members[0]["semantic"],
            "member_count": len(members),
            "members": [m["instance_id"] for m in members],
            "proposals": sorted({m["proposal_id"] for m in members}),
            "passed_count": len(passed_members),
            "rejected_count": len(members) - len(passed_members),
            "has_passed": bool(passed_members),
            "representative": representative["instance_id"],
            "representative_score": round(representative["score"], 4),
            "representative_passed": representative["passed"],
        })
    instance_to_group = {m: g["group_id"] for g in ordered for m in g["members"]}
    conflict_pairs = [
        {"group_a": instance_to_group[a], "group_b": instance_to_group[b],
         "instance_a": a, "instance_b": b, "semantic_pair": pair}
        for a, b, pair in conflicts
    ]
    return {
        "frame_id": evidence["frame_id"],
        "second": int(evidence["source"]["capture_time_s"]),
        "instances": len(instances),
        "groups_total": len(ordered),
        "groups_with_passed": sum(1 for g in ordered if g["has_passed"]),
        "groups_all_rejected": sum(1 for g in ordered if not g["has_passed"]),
        "singleton_groups": sum(1 for g in ordered if g["member_count"] == 1),
        "conflict_pairs": conflict_pairs,
        "groups": ordered,
    }


def aggregate_frames(frames):
    by_semantic = {}
    for frame in frames:
        for group in frame["groups"]:
            entry = by_semantic.setdefault(group["semantic"], {
                "instances": 0, "groups": 0, "groups_with_passed": 0})
            entry["instances"] += group["member_count"]
            entry["groups"] += 1
            if group["has_passed"]:
                entry["groups_with_passed"] += 1
    return {
        "frames": len(frames),
        "instances": sum(f["instances"] for f in frames),
        "groups_total": sum(f["groups_total"] for f in frames),
        "groups_with_passed": sum(f["groups_with_passed"] for f in frames),
        "groups_all_rejected": sum(f["groups_all_rejected"] for f in frames),
        "singleton_groups": sum(f["singleton_groups"] for f in frames),
        "conflict_pairs_total": sum(len(f["conflict_pairs"]) for f in frames),
        "by_semantic": {s: e for s, e in sorted(by_semantic.items())},
    }


def build_json_payload(run_dir, frames, totals, thresholds):
    return {
        "schema_version": "instance_groups_v1",
        "run": run_dir.name,
        "proposal_source": json.loads(
            (next(run_dir.glob("*/evidence.json"))).read_text(encoding="utf-8")
        ).get("proposal_source", ""),
        "thresholds": thresholds,
        "annotation_used_for_inference": False,
        "semantic_verified": False,
        "note": ("Derived object-group counting layer over raw segmentation "
                 "instances (same-semantic near-duplicate/containment union-find). "
                 "Groups are review statistics, not verified objects; nothing is "
                 "merged in the underlying run, deleted, or sem adjudicated."),
        "totals": totals,
        "frames": frames,
    }


def render_report(run_dir, frames, totals, thresholds):
    lines = []
    lines.append(f"instance grouping: {run_dir.name}")
    lines.append(f"frames={len(frames)}  duplicate_iou>={thresholds['duplicate_iou']}"
                 f"  containment=({thresholds['containment_fraction']},"
                 f" {thresholds['containment_area_ratio']})  same-semantic only")
    lines.append("")
    lines.append("[1] per-frame: instances -> groups (with passed / all rejected / singletons)")
    for frame in frames:
        lines.append(f"  {frame['second']:>2}s instances={frame['instances']}"
                     f" -> groups={frame['groups_total']}"
                     f" (passed={frame['groups_with_passed']}"
                     f" rejected={frame['groups_all_rejected']}"
                     f" singleton={frame['singleton_groups']})")
    lines.append("")
    lines.append(f"[2] totals: instances={totals['instances']} -> groups={totals['groups_total']}"
                 f" (with passed={totals['groups_with_passed']},"
                 f" all rejected={totals['groups_all_rejected']},"
                 f" singletons={totals['singleton_groups']})")
    lines.append("")
    lines.append("[3] per-semantic: instances -> groups (object-count proxy: groups with passed)")
    for semantic, entry in totals["by_semantic"].items():
        lines.append(f"  {semantic:<18} instances={entry['instances']:>4}"
                     f" -> groups={entry['groups']:>3}"
                     f" (with passed={entry['groups_with_passed']})")
    lines.append("")
    lines.append("[4] multi-member groups (review: same region proposed/segmented repeatedly)")
    for frame in frames:
        for group in frame["groups"]:
            if group["member_count"] > 1:
                lines.append(f"  {frame['second']:>2}s {group['group_id']} {group['semantic']}"
                             f" members={group['member_count']}"
                             f" passed={group['passed_count']} proposals={group['proposals']}"
                             f" rep={group['representative']}"
                             f" ({'passed' if group['representative_passed'] else 'rejected'},"
                             f" score={group['representative_score']})")
    lines.append("")
    lines.append(f"[5] cross-semantic conflict pairs among passed instances"
                 f" (IoU >= {thresholds['conflict_iou']}): {totals['conflict_pairs_total']}")
    for frame in frames:
        for pair in frame["conflict_pairs"]:
            lines.append(f"  {frame['second']:>2}s {pair['semantic_pair']}"
                         f" groups {pair['group_a']} x {pair['group_b']}"
                         f" ({pair['instance_a']} x {pair['instance_b']})")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Group near-duplicate instances per frame (read-only)")
    parser.add_argument("--run", type=Path, required=True,
                        help="Verified run directory with */evidence.json and masks.npz")
    parser.add_argument("--duplicate-iou", type=float, default=DUPLICATE_IOU)
    parser.add_argument("--containment-fraction", type=float, default=CONTAINMENT_FRACTION)
    parser.add_argument("--containment-area-ratio", type=float, default=CONTAINMENT_AREA_RATIO)
    parser.add_argument("--conflict-iou", type=float, default=CONFLICT_IOU)
    parser.add_argument("--output", type=Path, help="Optional text report file (UTF-8)")
    parser.add_argument("--json", type=Path, help="Optional machine-readable group file")
    args = parser.parse_args()

    run_dir = args.run.resolve(strict=True)
    frame_dirs = sorted(path.parent for path in run_dir.glob("*/evidence.json"))
    if not frame_dirs:
        raise SystemExit("No evidence.json frames found under --run")
    thresholds = {
        "duplicate_iou": args.duplicate_iou,
        "containment_fraction": args.containment_fraction,
        "containment_area_ratio": args.containment_area_ratio,
        "conflict_iou": args.conflict_iou,
    }
    frames = []
    for frame_dir in frame_dirs:
        frame = group_frame(frame_dir, args.duplicate_iou, args.containment_fraction,
                            args.containment_area_ratio, args.conflict_iou)
        frames.append(frame)
        print(f"grouped {frame['frame_id']}: {frame['instances']} instances"
              f" -> {frame['groups_total']} groups"
              f" ({frame['groups_with_passed']} with passed)", flush=True)
    frames.sort(key=lambda f: f["second"])
    totals = aggregate_frames(frames)
    text = render_report(run_dir, frames, totals, thresholds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json:
        write_json(args.json, build_json_payload(run_dir, frames, totals, thresholds))
        print(f"wrote group evidence to {args.json}")
    print(text)


if __name__ == "__main__":
    main()
