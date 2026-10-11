import argparse
from collections import Counter
import json
from pathlib import Path

from fuse_metric import confined_file
from run_offline import write_json


STAGES = ("IN_ROW", "APPROACH", "HEADLAND", "ENTER_ROW")
HAZARD_LABELS = {"person": "person", "vehicle": "vehicle", "tarp": "plastic_sheet_or_tarp", "pole": "pole"}


def evaluate(results, annotations):
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads(annotations.read_text(encoding="utf-8"))
    confusion = Counter()
    candidates = Counter()
    hazard_counts = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in HAZARD_LABELS}
    details = []
    correct = covered = uncertain = 0
    for item in manifest["frames"]:
        evidence = json.loads(confined_file(results, item["evidence_path"]).read_text(encoding="utf-8"))
        video = reference["videos"][item["video_name"]]
        second = item["sample_second"]
        if not isinstance(second, int) or not 0 <= second < len(video["stages"]):
            raise ValueError("Prediction sample_second does not match annotation timeline")
        actual = video["stages"][second]
        predicted = evidence["stage_prediction"]
        if actual not in STAGES or predicted not in (*STAGES, "UNKNOWN"):
            raise ValueError("Invalid navigation-stage label")
        confusion[(actual, predicted)] += 1
        covered += predicted != "UNKNOWN"
        correct += actual == predicted
        is_uncertain = second in video["uncertain_seconds"]
        uncertain += is_uncertain
        hypothesis = evidence["hypothesis"]["hypothesis"]
        candidates[(actual, hypothesis)] += 1
        present = {h["semantic"] for h in evidence["hazards"]}
        for name, annotation_name in HAZARD_LABELS.items():
            reference_present = any(
                h["type"] == annotation_name and h["t_start"] <= second <= h["t_end"]
                for h in video["hazards"]
            )
            detected = name in present
            key = "tp" if detected and reference_present else "fp" if detected else "fn" if reference_present else "tn"
            hazard_counts[name][key] += 1
        details.append({
            "frame_id": item["frame_id"], "reference_stage": actual,
            "prediction_stage": predicted, "reference_uncertain": is_uncertain,
            "candidate_hypothesis": hypothesis,
        })
    count = len(details)
    return {
        "schema_version": "field_boundary_evaluation_v1", "frame_count": count,
        "reference_reviewed_flag": reference["reviewed"], "uncertain_reference_frames": uncertain,
        "stage_coverage": covered / count if count else None,
        "stage_accuracy_on_covered_frames": correct / covered if covered else None,
        "stage_confusion": [{"reference": a, "prediction": p, "count": n} for (a, p), n in sorted(confusion.items())],
        "candidate_reference_crosstab": [{"reference": a, "hypothesis": h, "count": n} for (a, h), n in sorted(candidates.items())],
        "coarse_hazard_presence_counts": hazard_counts, "frames": details,
        "limitations": [
            "Selected development frames are not a held-out evaluation dataset",
            "Candidate headland/road hypotheses are not navigation-stage predictions",
            "Hazard interval annotations provide coarse presence references, not exhaustive per-frame pixel truth",
            "No pixel IoU, metric boundary accuracy or safe-maneuver success can be measured from these labels",
            "UNKNOWN abstentions are excluded from selective accuracy and explicitly counted in coverage",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only evaluation; labels never enter the inference process")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new evaluation output file")
    report = evaluate(args.results.resolve(strict=True), args.annotations.resolve(strict=True))
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("frame_count", "stage_coverage", "stage_accuracy_on_covered_frames")}))


if __name__ == "__main__":
    main()
