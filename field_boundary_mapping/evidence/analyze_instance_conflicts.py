"""Audit duplicate and conflicting segmentation instances in one verified run.

Read-only analysis over a run produced by run_proposals.py. For every frame it
loads evidence.json plus masks.npz and quantifies:
  [1] the instance census (proposals, instances, passed, rejected, flagged);
  [2] near-duplicate mask pairs (IoU >= duplicate threshold), split into
      same-proposal and cross-proposal pairs and by pass-status combination;
  [3] rejected instances that near-duplicate a spatially consistent instance
      (mask budget spent on twins of an accepted mask);
  [4] cross-semantic conflicts: two spatially consistent instances with
      different semantics overlapping above the conflict threshold;
  [5] whole/part containment pairs: a small mask almost fully inside a much
      larger one, which inflates per-object instance counts.
The audit supports human review and ablation planning only. It never edits
runs, never merges or deletes anything, and never marks semantics verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from field_boundary_mapping.proposals.run_offline import write_json

BASE = Path(__file__).resolve().parents[2]   # 仓库根

DUPLICATE_IOU = 0.95
CONFLICT_IOU = 0.5
CONTAINMENT_FRACTION = 0.95
CONTAINMENT_AREA_RATIO = 0.8
PASSED = "spatially_consistent"


def bbox_overlap_area(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, width) * max(0, height)


def pair_overlap(mask_a, mask_b):
    """Return IoU, contained-in-small fraction, areas, and which mask is smaller."""
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0 or area_a == 0 or area_b == 0:
        return {"iou": 0.0, "contained": 0.0, "small": 0, "large": 0, "small_is_a": True}
    iou = inter / (area_a + area_b - inter)
    small_is_a = area_a <= area_b
    small, large = (area_a, area_b) if small_is_a else (area_b, area_a)
    return {"iou": iou, "contained": inter / small, "small": small, "large": large,
            "small_is_a": small_is_a}


def load_frame(frame_dir):
    frame_dir = Path(frame_dir)
    evidence = json.loads((frame_dir / "evidence.json").read_text(encoding="utf-8"))
    with np.load(frame_dir / "masks.npz", allow_pickle=False) as archive:
        masks = {key: np.asarray(archive[key], dtype=bool) for key in archive.files}
    instances = []
    for item in evidence.get("instances", []):
        key = item["mask_key"]
        if key not in masks:
            raise ValueError(f"mask {key} missing from masks.npz")
        instances.append({
            "instance_id": item["instance_id"], "mask_key": key,
            "proposal_id": item["proposal_id"], "semantic": item["semantic"],
            "score": float(item["score"]), "bbox": list(item["bbox_xyxy"]),
            "passed": item["spatial_check"]["status"] == PASSED,
            "disagreement": bool(item.get("box_mask_disagreement")),
            "mask": masks[key],
        })
    return evidence, instances


def audit_frame(frame_dir, duplicate_iou=DUPLICATE_IOU, conflict_iou=CONFLICT_IOU,
                containment_fraction=CONTAINMENT_FRACTION,
                containment_area_ratio=CONTAINMENT_AREA_RATIO):
    evidence, instances = load_frame(frame_dir)
    duplicates, conflicts, containments = [], [], []
    best_rejected = {}
    count = len(instances)
    for i in range(count):
        a = instances[i]
        for j in range(i + 1, count):
            b = instances[j]
            if bbox_overlap_area(a["bbox"], b["bbox"]) <= 0:
                continue
            overlap = pair_overlap(a["mask"], b["mask"])
            iou, contained = overlap["iou"], overlap["contained"]
            if iou <= 0 and contained <= 0:
                continue
            cross_proposal = a["proposal_id"] != b["proposal_id"]
            same_semantic = a["semantic"] == b["semantic"]
            if a["passed"] and b["passed"]:
                status = "both_passed"
            elif not a["passed"] and not b["passed"]:
                status = "both_rejected"
            else:
                status = "one_passed"
            if iou >= duplicate_iou:
                duplicates.append({
                    "a": a["instance_id"], "b": b["instance_id"],
                    "a_proposal": a["proposal_id"], "b_proposal": b["proposal_id"],
                    "a_semantic": a["semantic"], "b_semantic": b["semantic"],
                    "iou": round(iou, 4), "status": status,
                    "cross_proposal": cross_proposal, "same_semantic": same_semantic,
                })
                if status == "one_passed":
                    rejected_one, passed_one = (a, b) if b["passed"] else (b, a)
                    current = best_rejected.get(rejected_one["instance_id"])
                    if current is None or iou > current["iou"]:
                        best_rejected[rejected_one["instance_id"]] = {
                            "iou": iou, "matched_passed": passed_one["instance_id"],
                            "matched_semantic": passed_one["semantic"],
                            "semantic": rejected_one["semantic"],
                        }
            if (contained >= containment_fraction
                    and overlap["small"] <= containment_area_ratio * overlap["large"]):
                small, large = (a, b) if overlap["small_is_a"] else (b, a)
                containments.append({
                    "small": small["instance_id"], "large": large["instance_id"],
                    "small_semantic": small["semantic"], "large_semantic": large["semantic"],
                    "contained_fraction": round(contained, 4),
                    "area_ratio": round(overlap["small"] / overlap["large"], 4),
                    "cross_proposal": cross_proposal, "status": status,
                })
            if a["passed"] and b["passed"] and not same_semantic and iou >= conflict_iou:
                conflicts.append({
                    "a": a["instance_id"], "b": b["instance_id"],
                    "a_proposal": a["proposal_id"], "b_proposal": b["proposal_id"],
                    "semantic_pair": "|".join(sorted((a["semantic"], b["semantic"]))),
                    "iou": round(iou, 4),
                })
    per_proposal = {}
    for instance in instances:
        per_proposal[instance["proposal_id"]] = per_proposal.get(instance["proposal_id"], 0) + 1
    passed = sum(1 for instance in instances if instance["passed"])
    redundant = [
        {"instance_id": instance_id, **{key: (round(value, 4) if key == "iou" else value)
                                         for key, value in record.items()}}
        for instance_id, record in sorted(best_rejected.items())
    ]
    return {
        "frame_id": evidence["frame_id"],
        "second": int(evidence["source"]["capture_time_s"]),
        "proposal_source": evidence.get("proposal_source", ""),
        "proposals": len(evidence.get("proposals", [])),
        "instances": count, "passed": passed, "rejected": count - passed,
        "flagged_disagreement": sum(1 for instance in instances if instance["disagreement"]),
        "max_instances_per_proposal": max(per_proposal.values(), default=0),
        "instances_per_proposal": per_proposal,
        "duplicate_pairs": duplicates,
        "duplicate_counts": {
            "total": len(duplicates),
            "cross_proposal": sum(1 for d in duplicates if d["cross_proposal"]),
            "same_proposal": sum(1 for d in duplicates if not d["cross_proposal"]),
            "both_passed": sum(1 for d in duplicates if d["status"] == "both_passed"),
            "one_passed": sum(1 for d in duplicates if d["status"] == "one_passed"),
            "both_rejected": sum(1 for d in duplicates if d["status"] == "both_rejected"),
            "cross_semantic": sum(1 for d in duplicates if not d["same_semantic"]),
        },
        "redundant_rejected": redundant,
        "conflicts": conflicts,
        "containment_pairs": containments,
    }


def _count_by(records, key):
    counts = {}
    for record in records:
        counts[record[key]] = counts.get(record[key], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def aggregate_reports(reports):
    duplicates = [d for r in reports for d in r["duplicate_pairs"]]
    conflicts = [c for r in reports for c in r["conflicts"]]
    containments = [c for r in reports for c in r["containment_pairs"]]
    redundant = [x for r in reports for x in r["redundant_rejected"]]
    return {
        "frames": len(reports),
        "proposals": sum(r["proposals"] for r in reports),
        "instances": sum(r["instances"] for r in reports),
        "passed": sum(r["passed"] for r in reports),
        "rejected": sum(r["rejected"] for r in reports),
        "flagged_disagreement": sum(r["flagged_disagreement"] for r in reports),
        "duplicate_counts": {
            key: sum(r["duplicate_counts"][key] for r in reports)
            for key in reports[0]["duplicate_counts"]
        },
        "duplicates_by_semantic_pair": _count_by(
            [{"pair": "|".join(sorted((d["a_semantic"], d["b_semantic"])))} for d in duplicates],
            "pair"),
        "conflicts_total": len(conflicts),
        "conflicts_by_semantic_pair": _count_by(conflicts, "semantic_pair"),
        "redundant_rejected_total": len(redundant),
        "redundant_by_semantic": _count_by(redundant, "semantic"),
        "containment_total": len(containments),
        "containment_by_semantic_pair": _count_by(
            [{"pair": "|".join(sorted((c["small_semantic"], c["large_semantic"])))}
             for c in containments], "pair"),
    }


def build_json_payload(run_dir, reports, totals, thresholds):
    return {
        "schema_version": "instance_conflicts_audit_v1",
        "run": run_dir.name,
        "proposal_source": reports[0]["proposal_source"],
        "thresholds": thresholds,
        "annotation_used_for_inference": False,
        "semantic_verified": False,
        "note": ("Duplicate/conflict statistics over raw segmentation instances. "
                 "Supports human review and ablation planning only; never merges, "
                 "deletes, or verifies anything."),
        "totals": totals,
        "frames": reports,
    }


def render_report(run_dir, reports, totals, thresholds):
    lines = []
    lines.append(f"instance duplicate/conflict audit: {run_dir.name}  source={reports[0]['proposal_source']}")
    lines.append(f"frames={len(reports)}  duplicate_iou>={thresholds['duplicate_iou']}"
                 f"  conflict_iou>={thresholds['conflict_iou']}"
                 f"  containment=({thresholds['containment_fraction']}, {thresholds['containment_area_ratio']})")
    lines.append("")
    lines.append("[1] per-frame census (proposals | instances | passed | rejected | flagged | max per proposal)")
    for r in reports:
        lines.append(f"  {r['second']:>2}s proposals={r['proposals']} instances={r['instances']}"
                     f" passed={r['passed']} rejected={r['rejected']}"
                     f" flagged={r['flagged_disagreement']} max_per_proposal={r['max_instances_per_proposal']}")
    lines.append("")
    dc = totals["duplicate_counts"]
    lines.append(f"[2] near-duplicate mask pairs (IoU >= {thresholds['duplicate_iou']})")
    lines.append(f"  totals: total={dc['total']} cross_proposal={dc['cross_proposal']}"
                 f" same_proposal={dc['same_proposal']} both_passed={dc['both_passed']}"
                 f" one_passed={dc['one_passed']} both_rejected={dc['both_rejected']}"
                 f" cross_semantic={dc['cross_semantic']}")
    pairs = " ".join(f"{name}={count}" for name, count in totals["duplicates_by_semantic_pair"].items())
    lines.append(f"  by semantic pair: {pairs or '-'}")
    for r in reports:
        rdc = r["duplicate_counts"]
        lines.append(f"  {r['second']:>2}s total={rdc['total']} cross={rdc['cross_proposal']}"
                     f" same={rdc['same_proposal']} one_passed={rdc['one_passed']}"
                     f" both_rejected={rdc['both_rejected']}")
    lines.append("")
    redundant = totals["redundant_rejected_total"]
    rejected = totals["rejected"]
    fraction = f"{100.0 * redundant / rejected:.1f}%" if rejected else "-"
    lines.append("[3] rejected instances near-duplicating a passed instance")
    lines.append(f"  total={redundant} of {rejected} rejected ({fraction})")
    by_semantic = " ".join(f"{name}={count}" for name, count in totals["redundant_by_semantic"].items())
    lines.append(f"  by rejected-instance semantic: {by_semantic or '-'}")
    lines.append("")
    lines.append(f"[4] cross-semantic conflicts among passed instances (IoU >= {thresholds['conflict_iou']})")
    lines.append(f"  total={totals['conflicts_total']}")
    by_pair = " ".join(f"{name}={count}" for name, count in totals["conflicts_by_semantic_pair"].items())
    lines.append(f"  by semantic pair: {by_pair or '-'}")
    for r in reports:
        for c in r["conflicts"]:
            lines.append(f"  {r['second']:>2}s {c['semantic_pair']} IoU={c['iou']}"
                         f" ({c['a']} x {c['b']}, proposals {c['a_proposal']}/{c['b_proposal']})")
    lines.append("")
    lines.append("[5] whole/part containment pairs"
                 f" (small >= {thresholds['containment_fraction']} inside,"
                 f" area <= {thresholds['containment_area_ratio']}x large)")
    lines.append(f"  total={totals['containment_total']}")
    by_pair = " ".join(f"{name}={count}" for name, count in totals["containment_by_semantic_pair"].items())
    lines.append(f"  by semantic pair: {by_pair or '-'}")
    for r in reports:
        for c in r["containment_pairs"]:
            lines.append(f"  {r['second']:>2}s {c['small_semantic']}|{c['large_semantic']}"
                         f" contained={c['contained_fraction']} area_ratio={c['area_ratio']}"
                         f" ({c['small']} in {c['large']}, status={c['status']})")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Duplicate/conflict audit for one verified run")
    parser.add_argument("--run", type=Path, required=True,
                        help="Verified run directory with */evidence.json and masks.npz")
    parser.add_argument("--duplicate-iou", type=float, default=DUPLICATE_IOU)
    parser.add_argument("--conflict-iou", type=float, default=CONFLICT_IOU)
    parser.add_argument("--containment-fraction", type=float, default=CONTAINMENT_FRACTION)
    parser.add_argument("--containment-area-ratio", type=float, default=CONTAINMENT_AREA_RATIO)
    parser.add_argument("--output", type=Path, help="Optional text report file (UTF-8)")
    parser.add_argument("--json", type=Path, help="Optional machine-readable audit file")
    args = parser.parse_args()

    run_dir = args.run.resolve(strict=True)
    frame_dirs = sorted(path.parent for path in run_dir.glob("*/evidence.json"))
    if not frame_dirs:
        raise SystemExit("No evidence.json frames found under --run")
    thresholds = {
        "duplicate_iou": args.duplicate_iou, "conflict_iou": args.conflict_iou,
        "containment_fraction": args.containment_fraction,
        "containment_area_ratio": args.containment_area_ratio,
    }
    reports = []
    for frame_dir in frame_dirs:
        report = audit_frame(frame_dir, args.duplicate_iou, args.conflict_iou,
                             args.containment_fraction, args.containment_area_ratio)
        reports.append(report)
        print(f"audited {report['frame_id']}: {report['instances']} instances,"
              f" {report['duplicate_counts']['total']} duplicate pairs,"
              f" {len(report['conflicts'])} conflicts", flush=True)
    reports.sort(key=lambda r: r["second"])
    totals = aggregate_reports(reports)
    text = render_report(run_dir, reports, totals, thresholds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json:
        write_json(args.json, build_json_payload(run_dir, reports, totals, thresholds))
        print(f"wrote audit evidence to {args.json}")
    print(text)


if __name__ == "__main__":
    main()
