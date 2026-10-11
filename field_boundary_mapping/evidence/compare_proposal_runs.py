"""Compare cloud-vlm probe (cloud_region_probe_v1) against local-2b probes (v3/v4).

Read-only development aid: prints per-frame instance tallies and box statistics so
local-vs-cloud proposal quality can be reviewed side by side. No aggregation into
accuracy claims; every status is spatial-only, semantics stay unverified.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[2] / "runs"   # 仓库根/runs


def frame_summary(run_name, frame_id):
    evidence = json.loads((BASE / run_name / frame_id / "evidence.json").read_text(encoding="utf-8"))
    instances = evidence["instances"]
    proposals = {p["proposal_id"]: p for p in evidence["proposals"]}
    statuses = {}
    for inst in instances:
        key = (inst["spatial_check"]["status"], bool(inst.get("box_mask_disagreement")))
        statuses[key] = statuses.get(key, 0) + 1
    lines = [f"{run_name}  source={evidence['proposal_source']}"]
    boxes = {}
    for inst in instances:
        pid = inst["proposal_id"]
        box = tuple(proposals[pid]["box_xyxy"])
        semantic = proposals[pid]["semantic"]
        boxes.setdefault(box, set()).add((pid, semantic))
    duplicates = {box: sorted(items) for box, items in boxes.items() if len(items) > 1}
    lines.append(f"  instances={len(instances)} proposals={len(proposals)}")
    for (status, flagged), count in sorted(statuses.items(), key=lambda kv: str(kv[0])):
        lines.append(f"  {status:<40} flagged={flagged} x{count}")
    lines.append(f"  identical boxes across different proposals: {len(duplicates)}")
    for box, items in duplicates.items():
        lines.append(f"    {list(box)} -> {items}")
    lines.append("  per-proposal:")
    for pid, p in proposals.items():
        matching = [i for i in instances if i["proposal_id"] == pid]
        scores = [round(i["score"], 2) for i in matching]
        statuses_p = [i["spatial_check"]["status"] for i in matching]
        lines.append(f"    {pid} {p['semantic']:<18} {str(p['box_xyxy']):<26} "
                     f"n={len(matching)} scores={scores} {statuses_p}")
    return "\n".join(lines)


def main():
    frame_ids = ["20260707193339_anv_video__f00000265",
                 "20260711083017_anv_video__f00000155",
                 "20260711090017_anv_video__f00000085"]
    for frame_id in frame_ids:
        print("=" * 100)
        print(frame_id)
        for run_name in ("vlm_region_probe_v4", "cloud_region_probe_v1"):
            print("-" * 100)
            print(frame_summary(run_name, frame_id))


if __name__ == "__main__":
    main()
