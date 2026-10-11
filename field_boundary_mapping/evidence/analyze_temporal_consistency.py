"""Cross-frame temporal consistency analysis for one verified region-proposal run.

Read-only: consumes frame evidence.json files produced by run_proposals.py plus the
per-second human annotation archive, and reports (1) a per-frame semantic timeline,
(2) IoU-greedy cross-frame chains with semantic-flip flags, and (3) agreement
between the hazard/event annotation timeline and the per-frame proposal presence.
Chains are built either over spatially-consistent instances (default) or over
per-frame object-group representatives (--chain-basis group_representatives,
reusing group_instances' same-semantic near-duplicate/containment grouping), so
that one chain tracks one physical object instead of its duplicate segmentation
instances. --json additionally writes per-unit chain evidence; that file supports
human semantic review only -- it never marks semantics verified and never filters
data. No network, no writes except the two optional output files.
"""
import argparse
import json
from pathlib import Path

from field_boundary_mapping.evidence.group_instances import (
    CONTAINMENT_AREA_RATIO, CONTAINMENT_FRACTION, DUPLICATE_IOU, group_frame,
)

BASE = Path(__file__).resolve().parents[2]   # 仓库根

HAZARD_SEMANTICS = {
    "vehicle": "vehicle",
    "plastic_sheet_or_tarp": "tarp",
    "person": "person",
    "water": "water",
    "ditch": "ditch",
}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_frames(run_dir):
    frames = []
    for evidence_path in sorted(run_dir.glob("*/evidence.json")):
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        second = int(evidence["source"]["capture_time_s"])
        proposals = evidence.get("proposals", [])
        instances = [
            {
                "instance_id": i["instance_id"],
                "semantic": i["semantic"],
                "bbox": i["bbox_xyxy"],
                "score": i["score"],
                "status": i["spatial_check"]["status"],
            }
            for i in evidence.get("instances", [])
        ]
        frames.append({
            "frame_id": evidence["frame_id"],
            "second": second,
            "capture_time_s": evidence["source"]["capture_time_s"],
            "proposal_source": evidence.get("proposal_source", ""),
            "proposals": proposals,
            "instances": instances,
        })
    frames.sort(key=lambda f: f["second"])
    return frames


def chain_units(frame):
    """Chainable units of a frame: attached object-group representatives in group
    mode, otherwise the frame's spatially-consistent instances."""
    if "chain_units" in frame:
        return frame["chain_units"]
    return [i for i in frame["instances"] if i["status"] == "spatially_consistent"]


def units_from_groups(frame, grouped):
    """Convert group_instances.group_frame output into chainable units.

    One unit per group containing at least one spatially-consistent instance,
    carrying that group's representative (its highest-scoring passed member).
    All-rejected groups never chain, mirroring the instance basis where rejected
    instances never join chains."""
    by_id = {i["instance_id"]: i for i in frame["instances"]}
    units = []
    for group in grouped["groups"]:
        if not group["has_passed"]:
            continue
        representative = by_id[group["representative"]]
        units.append({
            "instance_id": representative["instance_id"],
            "semantic": representative["semantic"],
            "bbox": representative["bbox"],
            "score": representative["score"],
            "group_id": group["group_id"],
            "member_count": group["member_count"],
            "passed_count": group["passed_count"],
        })
    return units


def attach_group_units(run_dir, frames, duplicate_iou, containment_fraction,
                       containment_area_ratio):
    """Group mode: attach per-frame object-group representatives as chain units.

    Reuses group_instances.group_frame (same-semantic near-duplicate/containment
    union-find over instance masks) so the chain layer and the grouping audit
    share one definition of an object group. The frame's raw instances stay
    untouched and keep feeding the semantic timeline and hazard agreement."""
    by_frame_id = {f["frame_id"]: f for f in frames}
    for frame_dir in sorted(path.parent for path in run_dir.glob("*/evidence.json")):
        grouped = group_frame(frame_dir, duplicate_iou, containment_fraction,
                              containment_area_ratio)
        frame = by_frame_id[grouped["frame_id"]]
        frame["chain_units"] = units_from_groups(frame, grouped)
    return frames


def build_chains(frames, threshold):
    """Greedy consecutive-frame IoU linking over chain units.

    Units are each frame's spatially-consistent instances, or its object-group
    representatives when the frame carries derived group units. Units are matched
    to active chains by box IoU (highest overlap first; a chain extends at most
    once per frame and a unit joins at most one chain). A chain that misses a
    frame is retired, so links exist only between consecutive sampled frames.
    Returns all chains in first-appearance order (single-frame chains included)
    plus a (frame_id, instance_id) -> chain_id linkage.
    """
    chains, active, linkage = [], [], {}
    for frame in frames:
        steps = [{"frame_id": frame["frame_id"], "second": frame["second"], **u}
                 for u in chain_units(frame)]
        pairs = []
        for chain_pos, chain in enumerate(active):
            for step_pos, step in enumerate(steps):
                overlap = iou(chain["steps"][-1]["bbox"], step["bbox"])
                if overlap >= threshold:
                    pairs.append((overlap, chain_pos, step_pos))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_chains, used_steps = set(), set()
        for overlap, chain_pos, step_pos in pairs:
            if chain_pos in used_chains or step_pos in used_steps:
                continue
            used_chains.add(chain_pos)
            used_steps.add(step_pos)
            step = steps[step_pos]
            active[chain_pos]["steps"].append(step)
            linkage[(step["frame_id"], step["instance_id"])] = active[chain_pos]["chain_id"]
        active = [chain for pos, chain in enumerate(active) if pos in used_chains]
        for step_pos, step in enumerate(steps):
            if step_pos not in used_steps:
                chain = {"chain_id": f"chain-{len(chains):02d}", "steps": [step]}
                chains.append(chain)
                active.append(chain)
                linkage[(step["frame_id"], step["instance_id"])] = chain["chain_id"]
    return chains, linkage


def chain_metrics(chain, sampled_seconds):
    steps = chain["steps"]
    semantics = [s["semantic"] for s in steps]
    counts = {}
    for semantic in semantics:
        counts[semantic] = counts.get(semantic, 0) + 1
    modal_semantic, modal_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {
        "chain_id": chain["chain_id"],
        "length": len(steps),
        "seconds": [s["second"] for s in steps],
        "semantic_sequence": semantics,
        "flip_count": sum(1 for a, b in zip(semantics, semantics[1:]) if a != b),
        "modal_semantic": modal_semantic,
        "modal_fraction": round(modal_count / len(steps), 4),
        "appearance_rate": round(len(steps) / len(sampled_seconds), 4),
    }


def semantic_run_stats(frames):
    """Per-semantic presence seconds and longest run over consecutive sampled frames.

    A semantic that never sustains across consecutive samples is a class-level
    instability signal (e.g. high-scoring intermittent ditch labels); it weighs on
    review priority, never removes anything.
    """
    summary = {}
    for frame in frames:
        present = {i["semantic"] for i in frame["instances"] if i["status"] == "spatially_consistent"}
        for semantic in present:
            summary.setdefault(semantic, {"present_seconds": [], "longest_consecutive_run": 0, "_run": 0})
    for semantic, info in summary.items():
        for frame in frames:
            present = any(i["semantic"] == semantic and i["status"] == "spatially_consistent"
                          for i in frame["instances"])
            if present:
                info["present_seconds"].append(frame["second"])
                info["_run"] += 1
                info["longest_consecutive_run"] = max(info["longest_consecutive_run"], info["_run"])
            else:
                info["_run"] = 0
        del info["_run"]
    return summary


def build_json_payload(run_dir, frames, chains, linkage, threshold,
                       basis="instances", grouping=None):
    sampled_seconds = [f["second"] for f in frames]
    metrics = {c["chain_id"]: chain_metrics(c, sampled_seconds) for c in chains}
    chain_entries = []
    for chain in chains:
        entry = dict(metrics[chain["chain_id"]])
        steps = []
        for s in chain["steps"]:
            step = {
                "frame_id": s["frame_id"], "second": s["second"], "instance_id": s["instance_id"],
                "semantic": s["semantic"], "score": round(s["score"], 4),
                "bbox_xyxy": [round(v, 1) for v in s["bbox"]],
            }
            if "group_id" in s:
                step.update({"group_id": s["group_id"], "member_count": s["member_count"],
                             "passed_count": s["passed_count"]})
            steps.append(step)
        entry["steps"] = steps
        chain_entries.append(entry)
    per_frame = []
    for frame in frames:
        entries = []
        for unit in chain_units(frame):
            chain_id = linkage[(frame["frame_id"], unit["instance_id"])]
            m = metrics[chain_id]
            entry = {
                "instance_id": unit["instance_id"], "semantic": unit["semantic"],
                "score": round(unit["score"], 4), "chain_id": chain_id,
                "chain_length": m["length"], "flip_count": m["flip_count"],
                "modal_semantic": m["modal_semantic"], "modal_fraction": m["modal_fraction"],
                "appearance_rate": m["appearance_rate"],
            }
            if "group_id" in unit:
                entry.update({"group_id": unit["group_id"], "member_count": unit["member_count"],
                              "passed_count": unit["passed_count"]})
            entries.append(entry)
        per_frame.append({"frame_id": frame["frame_id"], "second": frame["second"], "units": entries})
    if basis == "group_representatives":
        note = ("Cross-frame stability evidence over per-frame object-group representatives "
                "(groups containing at least one spatially-consistent instance; representative "
                "= highest-scoring passed member). Groups are derived review statistics, not "
                "verified objects. Supports human semantic review; never marks semantics "
                "verified, never filters or deletes candidates.")
    else:
        note = ("Cross-frame stability evidence over spatially-consistent instances only. "
                "Supports human semantic review; never marks semantics verified, never "
                "filters or deletes candidates.")
    payload = {
        "schema_version": "temporal_consistency_v2",
        "run": run_dir.name,
        "proposal_source": frames[0]["proposal_source"],
        "chain_basis": basis,
        "iou_threshold": threshold,
        "frames_total": len(frames),
        "sampled_seconds": sampled_seconds,
        "linkage": "consecutive_frame_iou_greedy",
        "annotation_used_for_inference": False,
        "semantic_verified": False,
        "note": note,
        "chains": chain_entries,
        "per_frame": per_frame,
        "semantic_summary": semantic_run_stats(frames),
    }
    if grouping is not None:
        payload["grouping"] = grouping
    return payload


def main():
    parser = argparse.ArgumentParser(description="Temporal consistency report for one verified run")
    parser.add_argument("--run", type=Path, required=True, help="Verified run directory with */evidence.json")
    parser.add_argument("--annotations", type=Path, default=BASE / "地头数据集标注_v0_20260919.json")
    parser.add_argument("--video", required=True, help="Video name to select annotation entry")
    parser.add_argument("--iou", type=float, default=0.2, help="Min IoU to link units across consecutive frames")
    parser.add_argument("--chain-basis", choices=("instances", "group_representatives"),
                        default="instances",
                        help="Build chains over passed instances (default) or over per-frame "
                             "object-group representatives (one chain per physical object)")
    parser.add_argument("--duplicate-iou", type=float, default=DUPLICATE_IOU,
                        help="Group mode: mask IoU above which same-semantic instances merge")
    parser.add_argument("--containment-fraction", type=float, default=CONTAINMENT_FRACTION,
                        help="Group mode: contained fraction for whole/part twin grouping")
    parser.add_argument("--containment-area-ratio", type=float, default=CONTAINMENT_AREA_RATIO,
                        help="Group mode: max small/large area ratio for containment grouping")
    parser.add_argument("--output", type=Path, help="Optional text report file (UTF-8)")
    parser.add_argument("--json", type=Path, help="Optional machine-readable chain evidence file")
    args = parser.parse_args()

    run_dir = args.run.resolve(strict=True)
    frames = load_frames(run_dir)
    if not frames:
        raise SystemExit("No evidence.json frames found under --run")
    grouping = None
    if args.chain_basis == "group_representatives":
        grouping = {
            "source": "group_instances.group_frame (same-semantic near-duplicate/containment union-find)",
            "duplicate_iou": args.duplicate_iou,
            "containment_fraction": args.containment_fraction,
            "containment_area_ratio": args.containment_area_ratio,
            "unit": "groups containing at least one spatially-consistent instance",
            "representative": "highest-scoring passed member of the group",
        }
        attach_group_units(run_dir, frames, args.duplicate_iou, args.containment_fraction,
                           args.containment_area_ratio)
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))["videos"][args.video]
    stages = annotation["stages"]
    events = annotation["events"]
    hazards = annotation["hazards"]

    lines = []
    lines.append(f"temporal consistency report: {run_dir.name}  source={frames[0]['proposal_source']}")
    lines.append(f"frames={len(frames)}  iou_threshold={args.iou}  chain_basis={args.chain_basis}")
    if grouping:
        lines.append(f"grouping: duplicate_iou>={args.duplicate_iou}"
                     f" containment=({args.containment_fraction}, {args.containment_area_ratio})"
                     " same-semantic only; unit = group with >=1 passed instance;"
                     " representative = highest-scoring passed member")
    lines.append("")

    lines.append("[1] per-frame semantic timeline (annotation stage | proposals | passed/verified instances)")
    for frame in frames:
        stage = stages[frame["second"]] if frame["second"] < len(stages) else "?"
        proposal_counts = {}
        for p in frame["proposals"]:
            proposal_counts[p["semantic"]] = proposal_counts.get(p["semantic"], 0) + 1
        passed_counts = {}
        for i in frame["instances"]:
            if i["status"] == "spatially_consistent":
                passed_counts[i["semantic"]] = passed_counts.get(i["semantic"], 0) + 1
        rejected = sum(1 for i in frame["instances"] if i["status"] != "spatially_consistent")
        proposals_text = ", ".join(f"{s}x{c}" for s, c in sorted(proposal_counts.items())) or "-"
        passed_text = ", ".join(f"{s}x{c}" for s, c in sorted(passed_counts.items())) or "-"
        suffix = ""
        if grouping:
            unit_counts = {}
            for unit in chain_units(frame):
                unit_counts[unit["semantic"]] = unit_counts.get(unit["semantic"], 0) + 1
            units_text = ", ".join(f"{s}x{c}" for s, c in sorted(unit_counts.items())) or "-"
            suffix = f"  G[{units_text}]"
        lines.append(f"  {frame['second']:>2}s {stage:<10} P[{proposals_text}]  V[{passed_text}]{suffix}  rej={rejected}")
    lines.append("")

    chains, linkage = build_chains(frames, args.iou)
    sampled_seconds = [f["second"] for f in frames]
    metrics = {c["chain_id"]: chain_metrics(c, sampled_seconds) for c in chains}
    lines.append("[2] cross-frame IoU chains over "
                 + ("object-group representatives" if grouping else "passed instances")
                 + " (semantic-agnostic matching)")
    chains_by_id = {c["chain_id"]: c for c in chains}
    multi = [m for m in metrics.values() if m["length"] >= 2]
    for m in sorted(multi, key=lambda m: (-m["length"], m["seconds"][0])):
        mark = "  <-- SEMANTIC FLIP" if m["flip_count"] else ""
        seq = " -> ".join(m["semantic_sequence"])
        extra = ""
        if grouping:
            members = sum(s.get("member_count", 1) for s in chains_by_id[m["chain_id"]]["steps"])
            extra = f"  instances={members}"
        lines.append(f"  len={m['length']:<2} {m['seconds'][0]}-{m['seconds'][-1]}s {seq}{mark}{extra}")
    singletons = sum(1 for m in metrics.values() if m["length"] == 1)
    flip_chains = sum(1 for m in multi if m["flip_count"])
    lines.append(f"  chains(len>=2)={len(multi)}  singletons={singletons}  with_semantic_flip={flip_chains}")
    if grouping:
        lines.append(f"  object tracks (all chains incl. singletons)={len(chains)}")
    summary = semantic_run_stats(frames)
    runs = " ".join(f"{s}={info['longest_consecutive_run']}" for s, info in
                    sorted(summary.items(), key=lambda kv: -kv[1]["longest_consecutive_run"]))
    lines.append(f"  semantic persistence (longest consecutive sampled-frame run): {runs}")
    lines.append("")

    lines.append("[3] hazard timeline agreement (annotation vs proposal presence at sampled seconds)")
    sampled = {f["second"] for f in frames}
    for hazard in hazards:
        semantic = HAZARD_SEMANTICS.get(hazard["type"], hazard["type"])
        span = range(hazard["t_start"], hazard["t_end"] + 1)
        in_span = sorted(s for s in sampled if s in span)
        entries = []
        for second in in_span:
            frame = next(f for f in frames if f["second"] == second)
            proposed = any(p["semantic"] == semantic for p in frame["proposals"])
            passed = any(i["semantic"] == semantic and i["status"] == "spatially_consistent"
                         for i in frame["instances"])
            entries.append(f"{second}s:{'P' if proposed else '-'}{'V' if passed else '-'}")
        lines.append(f"  {hazard['type']} (annotated {hazard['t_start']}-{hazard['t_end']}s, maps to '{semantic}')")
        lines.append(f"    sampled seconds in span: {' '.join(entries)}  (P=proposed, V=passed verification)")
    annotated_semantics = {HAZARD_SEMANTICS.get(h["type"], h["type"]) for h in hazards}
    other_semantics = sorted({p["semantic"] for f in frames for p in f["proposals"]} - annotated_semantics)
    if other_semantics:
        presence = {s: [f["second"] for f in frames if any(p["semantic"] == s for p in f["proposals"])]
                    for s in other_semantics}
        lines.append("  semantics proposed without an annotated hazard in this video (presence per sampled second):")
        for semantic in other_semantics:
            lines.append(f"    {semantic}: {presence[semantic]}")
    lines.append("")

    lines.append("[4] annotated event anchors vs neighboring sampled frames")
    for name, second in sorted(events.items()):
        if second is None:
            continue
        neighbors = [s for s in sorted(sampled) if abs(s - second) <= 2]
        lines.append(f"  {name}={second}s  sampled neighbors={neighbors}")
    lines.append("")

    report = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.json:
        payload = build_json_payload(run_dir, frames, chains, linkage, args.iou,
                                     basis=args.chain_basis, grouping=grouping)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote chain evidence to {args.json}")
    print(report)


if __name__ == "__main__":
    main()
