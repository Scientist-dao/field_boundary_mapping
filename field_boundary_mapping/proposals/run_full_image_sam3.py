"""Bypass VLM: run SAM3 with full-image surface prompts for every frame.

When the cloud VLM returns empty or malformed scenes, Stage 3 cannot run.
This script creates full-image regions for key surfaces and runs SAM3 directly,
producing the same overlay outputs as refine_scene_regions.py.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[2]   # 仓库根
sys.path.insert(0, str(BASE))

from field_boundary_mapping.proposals.run_offline import file_hash, write_image, write_json, VideoReader
from field_boundary_mapping.proposals.run_proposals import REGION_PROMPTS
from field_boundary_mapping.perception.sam3_local import LocalSAM3, offline_runtime
from field_boundary_mapping.semantic_pipeline import load_core, mask_contours
from field_boundary_mapping.scene.refine_scene_regions import (
    ROLE_PALETTE, EXTENT_CAVEAT, _polygon_mask, _overlaps, _fraction,
    _candidate_edge_diagnostics, render_overlays,
)
from field_boundary_mapping.proposals.run_proposals import assess_predictions
from field_boundary_mapping.scene.scene_understanding import GATES

RUN_SCHEMA = "field_functional_full_image_refinement_run_v1"
BACKEND = "local_sam3_full_image_surface_grounding"
PIPELINE_VERSION = "full-image-sam3-v1"

SURFACES_TO_QUERY = ["crop", "harvested_ground", "ground", "vehicle", "woody_vegetation"]
ROLE_MAP = {
    "crop": "current_field", "harvested_ground": "current_field",
    "ground": "farm_road", "vehicle": "obstacle", "woody_vegetation": "obstacle",
}


def load_frames_from_run(results_dir):
    results = Path(results_dir).resolve(strict=True)
    manifest = json.loads((results / "manifest.json").read_bytes())
    if manifest.get("schema_version") != "field_boundary_run_v1":
        raise ValueError("Source run must be a field_boundary_run_v1")
    frames = []
    for frame in manifest["frames"]:
        image_path = results / frame["image_path"]
        data = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {image_path}")
        frames.append({
            "frame": frame, "image": image, "image_bytes": data,
            "image_sha256": hashlib.sha256(data).hexdigest(),
        })
    return frames


def build_full_image_scene(width, height, surfaces):
    regions = []
    for idx, surface in enumerate(surfaces, 1):
        regions.append({
            "region_id": f"r{idx}", "role": ROLE_MAP.get(surface, "unknown"),
            "surface": surface, "polygon": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            "evidence": f"Full-image query for {surface}", "uncertainties": [],
        })
    return {
        "regions": regions, "relations": [], "boundaries": [],
        "unknowns": ["VLM bypassed; full-image regions used for SAM3 grounding"],
    }


def refine_full_image(image, scene, detections, core, frame_id):
    height, width = image.shape[:2]
    queries = []
    for region in scene["regions"]:
        if region["surface"] not in REGION_PROMPTS:
            continue
        queries.append({
            "proposal_id": region["region_id"], "semantic": region["surface"],
            "box_xyxy": [0, 0, width, height], "prompt": REGION_PROMPTS[region["surface"]],
            "relation": "unspecified", "rationale": region["evidence"],
        })
    assessed, instances, raw_masks = assess_predictions(image, queries, detections, frame_id, "full_image_bypass", core)
    coarse, candidates, functional_masks, regions = {}, {}, {}, []
    passed_unions = {}
    for region in scene["regions"]:
        region_id = region["region_id"]
        points = [[0, 0], [width, 0], [width, height], [0, height]]
        polygon = np.ones((height, width), dtype=bool)
        coarse[region_id] = polygon
        selected = [inst for inst in instances if inst["proposal_id"] == region_id]
        passed = [inst for inst in selected if inst["spatial_check"]["status"] == "spatially_consistent"]
        union = np.zeros((height, width), bool)
        raw_union = np.zeros_like(union)
        for inst in selected:
            raw_union |= inst["mask"]
            intersection = int(np.count_nonzero(inst["mask"] & polygon))
            inst.update(surface=region["surface"], role_hypothesis=region["role"], **GATES)
            inst["vlm_polygon_support"] = {
                "intersection_pixels": intersection,
                "mask_overlap_fraction": _fraction(intersection, int(inst["mask"].sum())),
                "polygon_support_fraction": _fraction(intersection, int(polygon.sum())),
            }
        for inst in passed:
            union |= inst["mask"]
        passed_unions[region_id] = union
        candidate = union.copy()
        edge_diagnostics = _candidate_edge_diagnostics(candidate, union)
        key = None
        if region["surface"] not in REGION_PROMPTS:
            status, reason = "unrefined_surface", "mixed_or_unknown_surface_has_no_sam_query"
        elif not selected:
            status, reason = "no_detection", "sam_returned_no_instance_for_this_region"
        elif not passed:
            status, reason = "spatial_rejected", "no_spatially_consistent_sam_instance"
        elif not candidate.any():
            status, reason = "spatial_rejected", "no_mask_support"
        else:
            status, reason = "refined_candidate", "full_image_sam3_surface_mask"
            key = "candidate_functional_" + region_id
            functional_masks[key] = candidate
            candidates[region_id] = candidate
        regions.append({
            **region, "coarse_polygon_px": points, "coarse_polygon_pixels": int(polygon.sum()),
            "status": status, "reason": reason,
            "raw_ids": [inst["instance_id"] for inst in selected],
            "pass_ids": [inst["instance_id"] for inst in passed],
            "passed_union_pixels": int(union.sum()), "candidate_pixels": int(candidate.sum()),
            "passed_union_overlap_fraction": _fraction(int(candidate.sum()), int(union.sum())),
            "coarse_polygon_support_fraction": _fraction(int(candidate.sum()), int(polygon.sum())),
            "raw_union_polygon_overlap_fraction": _fraction(int(np.count_nonzero(raw_union & polygon)), int(raw_union.sum()) if raw_union.sum() else 0),
            "candidate_functional_mask_key": key,
            "candidate_mask_contours": mask_contours(candidate) if key else [],
            "coarse_polygon_preserved": True, "segmentation_fallback_used": False,
            "quality_flags": (["polygon_clipping_edge_present"] if edge_diagnostics["polygon_cut_edge_pixels"] else []),
            **edge_diagnostics, "extent_caveat": EXTENT_CAVEAT, **GATES,
        })
    coarse_overlap = _overlaps(coarse, (height, width))
    candidate_overlap = _overlaps(candidates, (height, width))
    for region in regions:
        for name, overlap in (("coarse", coarse_overlap), ("candidate", candidate_overlap)):
            region[name + "_overlapping_region_ids"] = sorted({
                other for pair in overlap["pairs"] if region["region_id"] in pair["region_ids"]
                for other in pair["region_ids"] if other != region["region_id"]
            })
        if region["coarse_overlapping_region_ids"] or region["candidate_overlapping_region_ids"]:
            region["quality_flags"].append("functional_overlap_unresolved")
    raw_evidence = {
        "schema_version": "field_functional_raw_evidence_v1", "frame_id": frame_id,
        "queries": queries, "proposals": assessed["proposals"],
        "instances": [{k: v for k, v in inst.items() if k != "mask"} for inst in instances],
        "raw_mask_count": len(raw_masks), "all_returned_instances_retained": True,
        "mask_archive": "raw_masks.npz", "mask_clipping_applied": False,
        "instance_cap": None, **GATES,
    }
    evidence = {
        "schema_version": "field_functional_refinement_frame_v1", "frame_id": frame_id,
        "image": {"width": width, "height": height}, "regions": regions,
        "vlm_relations": scene["relations"], "vlm_boundaries": scene["boundaries"],
        "vlm_unknowns": scene["unknowns"], "raw_evidence": raw_evidence,
        "overlaps": {"coarse": coarse_overlap, "candidate": candidate_overlap},
        "boundary_geometry_diagnostics": [], "boundary_diagnostic_policy": "none",
        "boundary_support_diagnostics": [],
        "candidate_edge_diagnostic_policy": "Full-image polygons; no VLM clipping",
        "quality_flag_policy": "Geometry and unresolved overlap warnings only",
        "boundary_refinement_performed": False,
        "geometry": {"mode": "image_only", "distance_units": "pixels"},
        "display": {"default": "refined_overlay.jpg", "role_palette_rgb": ROLE_PALETTE,
                    "overlap_style": "black_white_stripes", "text_in_image": False,
                    "raw_evidence_path": "raw_evidence.json", "raw_masks_path": "raw_masks.npz"},
        "warnings": [EXTENT_CAVEAT, "VLM bypassed; full-image SAM3 surface queries only",
                     "Functional roles are hypotheses from surface type, not VLM semantics"],
        **GATES,
    }
    return evidence, raw_masks, functional_masks


def _resolve_by_priority(functional_masks, regions, raw_evidence):
    max_scores = {}
    for inst in raw_evidence.get("instances", []):
        pid = inst.get("proposal_id")
        score = inst.get("score", 0.0)
        if pid and score > max_scores.get(pid, 0.0):
            max_scores[pid] = score
    priority = np.full(functional_masks[list(functional_masks.keys())[0]].shape, -1.0)
    assignment = np.zeros_like(priority, dtype=np.int32)
    for region in regions:
        rid = region["region_id"]
        key = region.get("candidate_functional_mask_key")
        if not key or key not in functional_masks:
            continue
        score = max_scores.get(rid, 0.0)
        mask = functional_masks[key]
        better = mask & (score > priority)
        priority[better] = score
        assignment[better] = int(rid[1:])
    resolved = {}
    for region in regions:
        rid = region["region_id"]
        key = region.get("candidate_functional_mask_key")
        if key and key in functional_masks:
            resolved[rid] = assignment == int(rid[1:])
    return resolved


def _render_priority(image, resolved, regions):
    overlay = image.copy()
    counts = np.zeros(image.shape[:2], np.uint8)
    for mask in resolved.values():
        counts += mask.astype(np.uint8)
    for region in sorted(regions, key=lambda v: v["region_id"]):
        rid = region["region_id"]
        mask = resolved.get(rid)
        if mask is None or not mask.any():
            continue
        color = np.asarray(ROLE_PALETTE[region["role"]][::-1], dtype=np.uint8)
        visible = mask & (counts == 1)
        overlay[visible] = (0.5 * image[visible] + 0.5 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        edges = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(edges, contours, -1, 1, 1)
        overlay[(edges != 0) & (counts <= 1)] = color
    yy, xx = np.nonzero(counts > 1)
    if yy.size:
        stripes = np.where(((xx + yy) // 4) % 2 == 0, 245, 25).astype(np.uint8)
        overlay[yy, xx] = stripes[:, None]
    return overlay


def render_overlays_priority(image, evidence, functional_masks):
    resolved = _resolve_by_priority(functional_masks, evidence["regions"], evidence["raw_evidence"])
    overlays = {}
    overlays["refined_overlay.jpg"] = _render_priority(image, resolved, evidence["regions"])
    coarse_overlay = image.copy()
    for region in evidence["regions"]:
        color = np.asarray(ROLE_PALETTE[region["role"]][::-1], dtype=np.uint8)
        pts = np.asarray(region["coarse_polygon_px"], np.int32)
        cv2.polylines(coarse_overlay, [pts], True, color.tolist(), 2)
    overlays["coarse_overlay.jpg"] = coarse_overlay
    boundaries_overlay = image.copy()
    overlays["boundaries_overlay.jpg"] = boundaries_overlay
    return overlays


def run(args):
    with offline_runtime():
        results = Path(args.results).resolve(strict=True)
        output = Path(args.output).resolve()
        if output.exists():
            raise ValueError("Output directory already exists")
        output.mkdir(parents=True)

        print(f"Loading frames from {results.name}...", flush=True)
        frames = load_frames_from_run(results)
        print(f"Loaded {len(frames)} frames", flush=True)

        weights = Path(r"C:\Users\physi\Downloads\corn-auto-annotate\sam3.pt")
        if not weights.exists():
            raise FileNotFoundError(f"SAM3 weights not found at {weights}")
        clip_dir = weights.parent

        model = LocalSAM3(weights, clip_dir, args.confidence, args.device, args.threads)
        core = load_core()

        manifest = {
            "schema_version": RUN_SCHEMA, "configuration": {
                "source_results": str(results), "source_manifest_sha256": file_hash(results / "manifest.json"),
                "surfaces_queried": SURFACES_TO_QUERY, "pipeline_version": PIPELINE_VERSION,
                "backend": BACKEND, "surface_prompts": REGION_PROMPTS,
                "implementation_sha256": {Path(__file__).name: file_hash(__file__)},
            },
            "status": "running", "purpose": "development_vlm_bypass_full_image_sam3",
            "requested_frame_count": len(frames), "network_inference_performed": False,
            "extent_caveat": EXTENT_CAVEAT, **GATES, "frames": [],
        }
        write_json(output / "manifest.json", manifest)

        for idx, item in enumerate(frames, 1):
            frame, image, image_bytes = item["frame"], item["image"], item["image_bytes"]
            frame_id = frame["frame_id"]
            height, width = image.shape[:2]
            print(f"\n[{idx}/{len(frames)}] {frame_id} ({width}x{height})", flush=True)

            scene = build_full_image_scene(width, height, SURFACES_TO_QUERY)
            queries = []
            for region in scene["regions"]:
                if region["surface"] in REGION_PROMPTS:
                    queries.append({
                        "proposal_id": region["region_id"], "semantic": region["surface"],
                        "box_xyxy": [0, 0, width, height], "prompt": REGION_PROMPTS[region["surface"]],
                        "relation": "unspecified", "rationale": region["evidence"],
                    })

            t0 = time.perf_counter()
            detections, timing = model.predict_regions(image, queries) if queries else ([], {"status": "skipped", "total_seconds": 0.0})
            timing["total_seconds"] = time.perf_counter() - t0

            evidence, raw_masks, functional_masks = refine_full_image(image, scene, detections, core, frame_id)

            folder = output / frame_id
            folder.mkdir()
            original_name = "frame.jpg"
            write_image(folder / original_name, image)
            for name, overlay in render_overlays_priority(image, evidence, functional_masks).items():
                write_image(folder / name, overlay)
            np.savez_compressed(folder / "raw_masks.npz", **raw_masks)
            np.savez_compressed(folder / "functional_masks.npz", **functional_masks)
            write_json(folder / "raw_evidence.json", evidence["raw_evidence"])
            write_json(folder / "scene_evidence.json", evidence)

            manifest["frames"].append({
                "frame_id": frame_id, "image_path": original_name,
                "evidence_path": "scene_evidence.json", "raw_evidence_path": "raw_evidence.json",
                "raw_masks_path": "raw_masks.npz", "functional_masks_path": "functional_masks.npz",
                "coarse_overlay_path": "coarse_overlay.jpg", "refined_overlay_path": "refined_overlay.jpg",
                "boundaries_overlay_path": "boundaries_overlay.jpg",
                "region_count": len(scene["regions"]),
                "detection_count": len(detections),
                "candidate_count": sum(1 for r in evidence["regions"] if r["candidate_functional_mask_key"]),
                "timing_seconds": timing["total_seconds"],
            })
            manifest["status"] = "running"
            write_json(output / "manifest.json", manifest)
            print(f"  → {len(detections)} detections, {manifest['frames'][-1]['candidate_count']} candidates", flush=True)

        manifest["status"] = "completed"
        write_json(output / "manifest.json", manifest)
        print(f"\n✓ Completed: {output.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-image SAM3 surface grounding (VLM bypass)")
    parser.add_argument("--results", type=Path, required=True, help="Stage 1 run directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (must not exist)")
    parser.add_argument("--clip-dir", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
