import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from review_results import ResultStore
from run_offline import file_hash, write_image, write_json
from sam3_local import LocalSAM3, offline_runtime
from semantic_pipeline import COLORS, load_core, mask_contours


BASE = Path(__file__).resolve().parent
PROMPT_VERSION = "region-proposals-v2"
REGION_PROMPTS = {
    "crop": "agricultural crops",
    "harvested_ground": "harvested field",
    "ground": "bare soil",
    "water": "water",
    "ditch": "drainage ditch",
    "vehicle": "vehicle",
    "person": "person",
    "tarp": "plastic sheet",
    "woody_vegetation": "trees and bushes",
    "pole": "utility pole",
}
MAX_OUTSIDE_FRACTION = 0.15
# A rejected mask this confident and this large may mean the proposal box was too small,
# not that the segmentation is wrong; flag it for human review instead of silent folding.
DISAGREEMENT_SCORE = 0.4
DISAGREEMENT_AREA = 0.02


def fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(expected)}")


def identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError(f"Invalid {name}")


def prepare_inputs(results, proposals_path):
    with proposals_path.open("rb") as stream:
        raw = stream.read(262145)
    if len(raw) > 262144:
        raise ValueError("Proposal document exceeds 256 KiB")
    document = json.loads(raw)
    fields(document, ("schema_version", "proposal_source", "frames"), "proposal document")
    if document["schema_version"] != "field_boundary_proposals_v1":
        raise ValueError("Unsupported proposal schema")
    if document["proposal_source"] not in ("human_review", "assistant_visual_review", "local_vlm", "cloud_vlm"):
        raise ValueError("Unsupported proposal source; candidates must declare reviewed, local_vlm, or cloud_vlm provenance")
    if not isinstance(document["frames"], list) or not 1 <= len(document["frames"]) <= 16:
        raise ValueError("Select 1 to 16 frames explicitly")
    store = ResultStore(results)
    manifest_path = store.resolve("manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    frames = {}
    for frame in manifest["frames"]:
        identifier(frame.get("frame_id"), "source frame_id")
        if frame["frame_id"] in frames:
            raise ValueError("Duplicate source frame_id")
        frames[frame["frame_id"]] = frame
    prepared, seen = [], set()
    for record in document["frames"]:
        fields(record, ("frame_id", "image_sha256", "proposals"), "proposal frame")
        frame_id = record["frame_id"]
        identifier(frame_id, "frame_id")
        if frame_id in seen or frame_id not in frames:
            raise ValueError("Duplicate or unknown proposal frame_id")
        seen.add(frame_id)
        frame = frames[frame_id]
        image_bytes, _ = store.read_file(frame["image_path"])
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != record["image_sha256"] or digest != frame.get("artifact_sha256", {}).get(frame["image_path"]):
            raise ValueError("Proposal image hash does not match original image and source manifest")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] * image.shape[1] > 16777216:
            raise ValueError("Input must be a decodable image of at most 16777216 pixels")
        height, width = image.shape[:2]
        proposals = record["proposals"]
        if not isinstance(proposals, list) or not 1 <= len(proposals) <= 8:
            raise ValueError("Each frame requires 1 to 8 proposals")
        proposal_ids = set()
        queries = []
        for proposal in proposals:
            fields(proposal, ("proposal_id", "semantic", "box_xyxy", "relation", "rationale"), "proposal")
            identifier(proposal["proposal_id"], "proposal_id")
            if proposal["proposal_id"] in proposal_ids:
                raise ValueError("Duplicate proposal_id in frame")
            proposal_ids.add(proposal["proposal_id"])
            semantic = proposal["semantic"]
            if not isinstance(semantic, str) or semantic not in REGION_PROMPTS:
                raise ValueError("Unsupported proposed semantic")
            relation = proposal["relation"]
            if relation not in ("unspecified", "ego_candidate") or (relation == "ego_candidate" and semantic != "vehicle"):
                raise ValueError("Only vehicles may carry an unverified ego_candidate relation")
            box = proposal["box_xyxy"]
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError("Proposal box must be an in-image integer half-open xyxy rectangle")
            rationale = proposal["rationale"]
            if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
                raise ValueError("Proposal rationale must contain 1 to 500 characters")
            queries.append({**proposal, "prompt": REGION_PROMPTS[semantic]})
        prepared.append({"frame": frame, "image": image, "image_bytes": image_bytes,
                         "image_sha256": digest, "queries": queries})
    return document, prepared, hashlib.sha256(raw).hexdigest(), hashlib.sha256(manifest_bytes).hexdigest()


FLAGGED_CONTOUR = (0, 165, 255)  # BGR orange: high-score mask outside a likely too-small box


def cap_verified_detections(detections, passed, limit):
    """Ablation lever: keep the `limit` best detections per proposal after spatial
    verification, ranked by (spatially consistent first, score, original order).

    SAM3 scores do not track spatial consistency: on video 20260711090017 at 12.5s
    a ground proposal's three highest-scoring detections were one passed plus two
    rejected, while two lower-scoring passed detections were dropped by a pure
    score cap. Passed instances are therefore protected first; the audit records
    how many passed instances (if any) were still dropped, never silently.
    Ties are broken by original order.
    Returns (kept_detections_in_original_order, audit).
    """
    if limit is None:
        return detections, None
    if limit < 1:
        raise ValueError("Instance cap must be a positive integer")
    by_proposal = {}
    for index, detection in enumerate(detections):
        by_proposal.setdefault(detection.get("proposal_id"), []).append(index)
    kept_indexes = set()
    dropped, dropped_passed = {}, 0
    for proposal_id, indexes in by_proposal.items():
        ranked = sorted(indexes, key=lambda i: (not passed[i], -detections[i]["score"], i))
        kept_indexes.update(ranked[:limit])
        if len(indexes) > limit:
            dropped[proposal_id] = len(indexes) - limit
            dropped_passed += sum(1 for i in ranked[limit:] if passed[i])
    kept = [d for index, d in enumerate(detections) if index in kept_indexes]
    audit = {"limit": limit, "dropped_per_proposal": dropped,
             "dropped_total": sum(dropped.values()), "dropped_passed_total": dropped_passed}
    return kept, audit



def render_candidates(image, instances, queries, clean=False):
    """Render candidate overlays.

    clean=True is the review view: spatially consistent instances (white contour) plus
    rejected masks flagged box_mask_disagreement (orange contour — the proposal box may
    be too small, so the mask is shown for human review instead of folded away);
    low-confidence rejected noise is dropped. The full overlay keeps every raw mask;
    the *_passed.jpg files are a presentation variant only — masks.npz and evidence.json
    are never filtered.
    """
    selected = [instance for instance in instances if not clean
                or instance["spatial_check"]["status"] == "spatially_consistent"
                or instance.get("box_mask_disagreement") is True]
    overlay = image.copy()
    for instance in selected:
        color = COLORS.get(instance["semantic"], (188, 119, 63))[::-1]
        mask = instance["mask"]
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
    for instance in selected:
        contours, _ = cv2.findContours(instance["mask"].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if instance["spatial_check"]["status"] == "spatially_consistent":
            color = (255, 255, 255)
        elif instance.get("box_mask_disagreement") is True:
            color = FLAGGED_CONTOUR
        else:
            color = (35, 35, 240)
        cv2.drawContours(overlay, contours, -1, color, 2)
    for query in queries:
        x0, y0, x1, y1 = query["box_xyxy"]
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 2)
    return overlay


def assess_predictions(image, queries, detections, frame_id, proposal_source, core):
    by_id = {query["proposal_id"]: query for query in queries}
    instances, arrays = [], {}
    for index, detection in enumerate(detections):
        query = by_id.get(detection.get("proposal_id"))
        if (query is None or detection["semantic"] != query["semantic"] or detection["prompt"] != query["prompt"]
                or detection.get("box_xyxy") != query["box_xyxy"]):
            raise ValueError("Detection does not match its bound proposal")
        raw_mask = np.asarray(detection["mask"])
        if raw_mask.dtype != np.bool_ or raw_mask.shape != image.shape[:2]:
            raise ValueError("Detection mask must be boolean and match the original image")
        mask = np.ascontiguousarray(raw_mask)
        score = detection["score"]
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detection score must be a finite fraction")
        spatial = core.assess_region_support(mask, query["box_xyxy"], MAX_OUTSIDE_FRACTION)
        mask_key = f"candidate_{index:04d}"
        arrays[mask_key] = mask
        area = float(mask.mean())
        rejected = spatial["status"] != "spatially_consistent"
        instances.append({
            "instance_id": mask_key, "mask_key": mask_key, "proposal_id": query["proposal_id"],
            "semantic": query["semantic"], "prompt": query["prompt"],
            "score": score, "bbox_xyxy": detection["bbox_xyxy"],
            "area_fraction": area, "contours": mask_contours(mask),
            "evidence_frame_id": frame_id, "evidence_group": frame_id,
            "relation_hypothesis": query["relation"], "ego_vehicle_verified": False,
            "suppressed_as_ego": False, "semantic_verified": False,
            "status": "spatial_rejected" if rejected else "spatial_candidate_semantics_unverified",
            "box_mask_disagreement": rejected and score >= DISAGREEMENT_SCORE and area >= DISAGREEMENT_AREA,
            "spatial_check": spatial, "mask": mask,
        })
    proposal_records = []
    for query in queries:
        selected = [instance for instance in instances if instance["proposal_id"] == query["proposal_id"]]
        consistent = sum(instance["spatial_check"]["status"] == "spatially_consistent" for instance in selected)
        proposal_records.append({**query, "proposal_source": proposal_source,
                                 "detection_count": len(selected), "spatially_consistent_count": consistent,
                                 "status": "no_detection" if not selected else "spatial_candidate" if consistent else "spatial_rejected",
                                 "instance_ids": [instance["instance_id"] for instance in selected],
                                 "semantic_verified": False, "ego_vehicle_verified": False})
    public = [{key: value for key, value in instance.items() if key != "mask"} for instance in instances]
    evidence = {
        "schema_version": "field_boundary_frame_v1", "frame_id": frame_id,
        "image": {"width": image.shape[1], "height": image.shape[0]},
        "stage_prediction": "UNKNOWN", "stage_reason": "Reviewed region proposals are not automatic navigation-stage predictions",
        "hypothesis": {"hypothesis": "unknown", "status": "unknown", "reasons": ["region_semantics_unverified"],
                       "maneuver_verified": False, "authorization": "unknown"},
        "geometry": {"mode": "image_only", "missing": ["synchronized_pose", "camera_calibration", "metric_depth"]},
        "regions": [instance for instance in public if instance["semantic"] not in ("vehicle", "person", "ditch", "tarp", "pole")],
        "hazards": [instance for instance in public if instance["semantic"] in ("vehicle", "person", "ditch", "tarp", "pole")],
        "instances": public, "proposals": proposal_records,
        "proposal_source": proposal_source, "region_proposals_used": True,
        "unknown_fraction": 1.0, "conflict_fraction": None,
        "unknown_reason": "All region semantics remain hypotheses; no verified surface labels are produced",
        "boundaries": [], "world_map": None, "annotation_used_for_inference": False,
        "warnings": [
        ("VLM boxes guide segmentation; the VLM is not verified ground truth and this is not a held-out accuracy test"
         if proposal_source in ("local_vlm", "cloud_vlm") else
         "Reviewed boxes guide segmentation; this is not autonomous VLM perception or a held-out accuracy test"),
        "Spatial consistency does not establish correct semantics; raw rejected masks are retained, not clipped",
        "Ego relation is only a proposal; vehicle detections are never suppressed",
        "All prompts in a frame share one evidence group, not independent observations",
        "box_mask_disagreement flags high-scoring large rejected masks: the proposal box may be"
        " too small; it is a review hint, not a semantic verification",
    ],
    }
    return evidence, instances, arrays


def run(args):
    document, prepared, proposals_hash, source_manifest_hash = prepare_inputs(args.results, args.proposals)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Proposal output must be a new directory; existing results are never overwritten")
    core = load_core()
    weights = args.weights.resolve(strict=True)
    configuration = {
        "pipeline_version": PROMPT_VERSION, "weights_sha256": file_hash(weights),
        "confidence": args.confidence, "device": args.device, "threads": args.threads,
        "proposals_sha256": proposals_hash, "source_manifest_sha256": source_manifest_hash,
        "region_prompts": REGION_PROMPTS, "max_outside_fraction": MAX_OUTSIDE_FRACTION,
        "disagreement_score": DISAGREEMENT_SCORE, "disagreement_area": DISAGREEMENT_AREA,
        "max_instances_per_proposal": getattr(args, "max_instances_per_proposal", None),
        "implementation_sha256": {path.name: file_hash(path) for path in
                                  (Path(__file__), BASE / "sam3_local.py", BASE / "semantic_pipeline.py",
                                   BASE / "review_results.py", BASE / "run_offline.py", Path(core.__file__))},
    }
    model = LocalSAM3(weights, args.clip_dir, args.confidence, args.device, args.threads)
    output.mkdir(parents=True)
    manifest = {"schema_version": "field_boundary_run_v1", "configuration": configuration,
                "status": "running", "requested_frame_count": len(prepared),
                "purpose": "reviewed_region_proposal_development_not_autonomous_vlm_or_held_out_test",
                "proposal_source": document["proposal_source"], "region_proposals_used": True,
                "annotation_used_for_inference": False, "frames": []}
    write_json(output / "proposals.json", document)
    write_json(output / "manifest.json", manifest)
    for item in prepared:
        frame, image, queries = item["frame"], item["image"], item["queries"]
        frame_id = frame["frame_id"]
        print(f"region inference {frame_id}; {len(queries)} proposals", flush=True)
        started = time.perf_counter()
        detections, timing = model.predict_regions(image, queries)
        evidence, instances, arrays = assess_predictions(image, queries, detections, frame_id, document["proposal_source"], core)
        cap = getattr(args, "max_instances_per_proposal", None)
        cap_audit = None
        if cap is not None:
            passed = [instance["spatial_check"]["status"] == "spatially_consistent" for instance in instances]
            kept, cap_audit = cap_verified_detections(detections, passed, cap)
            if len(kept) != len(detections):
                evidence, instances, arrays = assess_predictions(image, queries, kept, frame_id, document["proposal_source"], core)
            evidence["instance_cap"] = cap_audit
        evidence.update({
            "source": {key: frame[key] for key in ("video_name", "video_sha256", "frame_index", "capture_time_s")},
            "input_image_sha256": item["image_sha256"], "proposals_sha256": proposals_hash,
            "timing": timing,
            "model": {**model.metadata, "backend": "local_sam3_reviewed_region_proposals", "vlm_used": False,
                      "prompt_version": PROMPT_VERSION, "prompts": queries, "weights_sha256": configuration["weights_sha256"]},
        })
        folder = output / frame_id
        folder.mkdir()
        original_name = "frame" + Path(frame["image_path"]).suffix.lower()
        (folder / original_name).write_bytes(item["image_bytes"])
        write_image(folder / "overlay.jpg", render_candidates(image, instances, queries))
        write_image(folder / "overlay_passed.jpg", render_candidates(image, instances, queries, clean=True))
        views = []
        for query in queries:
            filename = f"proposal_{query['proposal_id']}.jpg"
            passed_name = f"proposal_{query['proposal_id']}_passed.jpg"
            selected = [instance for instance in instances if instance["proposal_id"] == query["proposal_id"]]
            write_image(folder / filename, render_candidates(image, selected, [query]))
            write_image(folder / passed_name, render_candidates(image, selected, [query], clean=True))
            views.append({"proposal_id": query["proposal_id"], "label": f"{query['proposal_id']} · {query['semantic']}",
                          "overlay_path": f"{frame_id}/{filename}",
                          "overlay_passed_path": f"{frame_id}/{passed_name}"})
        np.savez_compressed(folder / "masks.npz", **arrays)
        write_json(folder / "evidence.json", evidence)
        exported = {key: frame[key] for key in ("frame_id", "video_name", "video_sha256", "frame_index", "capture_time_s")}
        exported.update({"image_path": f"{frame_id}/{original_name}", "overlay_path": f"{frame_id}/overlay.jpg",
                         "overlay_passed_path": f"{frame_id}/overlay_passed.jpg",
                         "evidence_path": f"{frame_id}/evidence.json", "masks_path": f"{frame_id}/masks.npz", "proposal_views": views})
        artifacts = [exported[key] for key in ("image_path", "overlay_path", "overlay_passed_path", "evidence_path", "masks_path")]
        artifacts.extend(view["overlay_path"] for view in views)
        artifacts.extend(view["overlay_passed_path"] for view in views)
        exported["artifact_sha256"] = {relative: file_hash(output / relative) for relative in artifacts}
        manifest["frames"].append(exported)
        write_json(output / "manifest.json", manifest)
        cap_note = f"; cap {cap_audit['dropped_total']} dropped" if cap_audit else ""
        print(f"saved {frame_id}; {len(instances)} raw masks{cap_note}; {time.perf_counter() - started:.1f}s", flush=True)
    manifest["status"] = "completed"
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Local SAM3 region proposals with C++ spatial checks; no VLM or control")
    parser.add_argument("--results", type=Path, required=True, help="Existing source run with hashed original images")
    parser.add_argument("--proposals", type=Path, required=True, help="Explicit reviewed boxes bound to original image hashes")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--max-instances-per-proposal", type=int, default=None,
                        help="Ablation lever: keep only the N best detections per proposal, ranked after spatial verification (passed instances first, then score; default: off)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with offline_runtime():
        run(args)


if __name__ == "__main__":
    main()
