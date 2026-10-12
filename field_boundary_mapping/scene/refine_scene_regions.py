"""Offline surface grounding of archived functional hypotheses, not boundary verification."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from field_boundary_mapping.proposals.run_offline import file_hash, write_image, write_json
from field_boundary_mapping.proposals.run_proposals import MAX_OUTSIDE_FRACTION, REGION_PROMPTS, assess_predictions, identifier
from field_boundary_mapping.scene.run_scene_understanding import load_scene_run
from field_boundary_mapping.perception.sam3_local import LocalSAM3, offline_runtime
from field_boundary_mapping.scene.scene_understanding import (GATES, _intersects, full_width_box, parse_scene, pixel_points,
                                 region_geometry_kind, region_outline, scene_queries)
from field_boundary_mapping.semantic_pipeline import load_core, mask_contours


BASE = Path(__file__).resolve().parents[2]   # 仓库根
PKG = BASE / "field_boundary_mapping"        # Python 包根
PIPELINE_VERSION = "functional-scene-sam3-v3"
RUN_SCHEMA = "field_functional_refinement_run_v1"
BACKEND = "local_sam3_functional_scene_grounding"
EXTENT_CAVEAT = (
    "Candidate functional masks are unions of spatially consistent SAM surface masks "
    "INTERSECTED with coarse VLM localization (a polygon or an axis-aligned box). "
    "Their extent is clipped by VLM localization; resulting contours are not observed "
    "or verified physical boundaries."
)
# RGB, independent of the surface-class palette. Legends live in JSON, never on images.
ROLE_PALETTE = {
    "current_field": (0, 114, 178), "neighbor_field": (0, 158, 115),
    "headland_candidate": (230, 159, 0), "farm_road": (204, 121, 167),
    "obstacle": (213, 55, 0), "unknown": (175, 175, 175),
}


def _polygon_mask(points, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, np.int32)], 1)
    return mask.astype(bool)


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _overlaps(masks, shape):
    counts = np.zeros(shape, np.uint8)
    pairs = []
    entries = sorted(masks.items())
    for index, (region_id, mask) in enumerate(entries):
        counts += mask
        for other_id, other in entries[index + 1:]:
            pixels = int(np.count_nonzero(mask & other))
            if pixels:
                pairs.append({"region_ids": [region_id, other_id], "overlap_pixels": pixels,
                              "region_overlap_fractions": [pixels / int(mask.sum()), pixels / int(other.sum())]})
    pixels = int(np.count_nonzero(counts > 1))
    return {"has_overlap": bool(pixels), "overlap_pixels": pixels,
            "overlap_fraction": pixels / counts.size, "pairs": pairs,
            "resolution": "unresolved_independent_region_layers"}


def _simple_open_line(points):
    if len(points) < 2 or len(set(map(tuple, points))) != len(points):
        return False
    edges = list(zip(points, points[1:]))  # Never close the VLM boundary polyline.
    for index, (a, b) in enumerate(edges):
        if index + 1 < len(edges):
            c = edges[index + 1][1]
            u, v = np.subtract(b, a), np.subtract(c, b)
            if u[0] * v[1] == u[1] * v[0] and np.dot(u, v) <= 0:
                return False
        for c, d in edges[index + 2:]:
            if _intersects(a, b, c, d):
                return False
    return True


def _boundary_diagnostics(boundaries, candidates, width, height):
    """Conservative one-way distances; no snapping, closing, or boundary replacement."""
    contours_by_id = {}
    for region_id, mask in candidates.items():
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # Only this easily established simple geometry is supported; omit everything else.
        if (hierarchy is not None and len(contours) == 1 and len(contours[0]) >= 3
                and cv2.contourArea(contours[0]) > 0 and cv2.isContourConvex(contours[0])):
            contours_by_id[region_id] = contours[0]
    diagnostics = []
    for boundary in boundaries:
        points = pixel_points(boundary["points"], width, height)
        if not _simple_open_line(points):
            continue
        line = np.asarray(points, dtype=float)
        lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
        spacing = max(1.0, float(lengths.sum()) / 4096)
        samples = []
        for a, b, length in zip(line, line[1:], lengths):
            steps = max(1, math.ceil(float(length) / spacing))
            samples.extend(a + (b - a) * t / steps for t in range(steps))
        samples.append(line[-1])
        for region_id in (boundary["side_a"], boundary["side_b"]):
            contour = contours_by_id.get(region_id)
            if contour is None:
                continue
            distances = [abs(cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)) for p in samples]
            diagnostics.append({
                "boundary_id": boundary["boundary_id"], "region_id": region_id,
                "status": "geometry_diagnostic_only", "units": "image_pixels",
                "method": "sampled_open_polyline_to_candidate_contour_one_way_unsigned_distance",
                "sample_count": len(samples), "maximum_sample_spacing_px": spacing,
                "minimum_distance_px": min(distances), "mean_distance_px": float(np.mean(distances)),
                "maximum_distance_px": max(distances), "boundary_refinement_performed": False,
                "contour_extent": "clipped_by_coarse_vlm_polygon", **GATES,
            })
    return diagnostics


def _interior_3x3(mask):
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _inner_edges(mask):
    """Foreground-side, 8-neighbor one-pixel edges, excluding the image frame."""
    edges = mask & ~_interior_3x3(mask)
    border_pixels = int(edges.sum())
    edges[[0, -1], :] = False
    edges[:, [0, -1]] = False
    return edges, border_pixels - int(edges.sum())


def _candidate_edge_diagnostics(candidate, union):
    edges, border_pixels = _inner_edges(candidate)
    count = int(edges.sum())
    cut_count = int(np.count_nonzero(edges & _interior_3x3(union)))
    return {"candidate_edge_pixels": count, "polygon_cut_edge_pixels": cut_count,
            "polygon_cut_edge_fraction": _fraction(cut_count, count),
            "image_border_edge_pixels": border_pixels}


def _boundary_support_diagnostics(boundaries, passed_unions, width, height):
    """One-way raster distances to unclipped SAM edges; not semantic verification."""
    diagnostics, by_region = [], {}
    threshold = max(2.0, 0.005 * math.hypot(width, height))
    for boundary in boundaries:
        points = pixel_points(boundary["points"], width, height)
        degenerate = len(set(map(tuple, points))) < 2
        raster = np.zeros((height, width), np.uint8)
        if not degenerate:
            cv2.polylines(raster, [np.asarray(points, np.int32)], False, 1, 1, cv2.LINE_8)
        yy, xx = np.nonzero(raster)
        border_count = int(np.count_nonzero((xx == 0) | (xx == width - 1) | (yy == 0) | (yy == height - 1)))
        for side in ("side_a", "side_b"):
            region_id = boundary[side]
            if region_id == "unknown":  # Not a named region; every named side gets a record.
                continue
            record = {
                "boundary_id": boundary["boundary_id"], "side": side, "region_id": region_id,
                "status": "degenerate_polyline" if degenerate else "geometry_diagnostic_only",
                "units": "image_pixels",
                "method": "open_polyline_unique_1px_raster_to_unclipped_sam_inner_edge_distance_transform_l2",
                "edge_extent": "box_passed_raw_union_unclipped_image_border_excluded",
                "sample_count": int(xx.size), "threshold_px": threshold,
                "mean_distance_px": None, "maximum_distance_px": None, "supported_fraction": None,
                "image_border_sample_fraction": _fraction(border_count, int(xx.size)),
                "boundary_refinement_performed": False, **GATES,
            }
            diagnostics.append(record)
            if not degenerate:
                by_region.setdefault(region_id, []).append((record, yy, xx))
    # One distance transform per named region, without retaining all full-frame transforms.
    for region_id, requests in by_region.items():
        union = passed_unions.get(region_id)
        if union is None:
            status = "missing_passed_union"
        elif not union.any():
            status = "empty_passed_union"
        else:
            edges, _ = _inner_edges(union)
            status = "geometry_diagnostic_only" if edges.any() else "no_interior_edges"
        if status != "geometry_diagnostic_only":
            for record, _, _ in requests:
                record["status"] = status
            continue
        distances = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for record, yy, xx in requests:
            samples = distances[yy, xx]
            record.update(mean_distance_px=float(np.mean(samples)), maximum_distance_px=float(np.max(samples)),
                          supported_fraction=float(np.mean(samples <= threshold)))
    return diagnostics


def refine_regions(image, scene, detections, core, frame_id):
    """Return evidence, raw masks and independent candidate masks, without I/O."""
    identifier(frame_id, "frame_id")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1
            or image.shape[0] * image.shape[1] > 16777216):
        raise ValueError("Expected a nonempty uint8 BGR image <= 16777216 pixels")
    scene = parse_scene(json.dumps(scene, allow_nan=False))
    height, width = image.shape[:2]
    queries = scene_queries(scene, width, height)
    assessed, instances, raw_masks = assess_predictions(image, queries, detections, frame_id, "cloud_vlm", core)
    coarse, candidates, functional_masks, regions = {}, {}, {}, []
    passed_unions = {}
    for region in scene["regions"]:
        region_id = region["region_id"]
        points = region_outline(region, width, height)
        polygon = _polygon_mask(points, (height, width))
        coarse[region_id] = polygon
        selected = [instance for instance in instances if instance["proposal_id"] == region_id]
        passed = [instance for instance in selected if instance["spatial_check"]["status"] == "spatially_consistent"]
        union = np.zeros((height, width), bool)
        raw_union = np.zeros_like(union)
        for instance in selected:
            raw_union |= instance["mask"]
            intersection = int(np.count_nonzero(instance["mask"] & polygon))
            instance.update(surface=region["surface"], role_hypothesis=region["role"], **GATES)
            instance["vlm_polygon_support"] = {
                "intersection_pixels": intersection,
                "mask_overlap_fraction": _fraction(intersection, int(instance["mask"].sum())),
                "polygon_support_fraction": _fraction(intersection, int(polygon.sum())),
            }
        for instance in passed:
            union |= instance["mask"]
        passed_unions[region_id] = union
        candidate = union & polygon
        edge_diagnostics = _candidate_edge_diagnostics(candidate, union)
        key = None
        if region["surface"] not in REGION_PROMPTS:
            status, reason = "unrefined_surface", "mixed_or_unknown_surface_has_no_sam_query"
        elif not selected:
            status, reason = "no_detection", "sam_returned_no_instance_for_this_region"
        elif not passed:
            status, reason = "spatial_rejected", "no_spatially_consistent_sam_instance"
        elif not candidate.any():
            status, reason = "spatial_rejected", "box_consistent_masks_have_no_polygon_support"
        else:
            status, reason = "refined_candidate", "spatially_consistent_union_intersected_with_coarse_vlm_polygon"
            key = "candidate_functional_" + region_id
            functional_masks[key] = candidate
            candidates[region_id] = candidate
        quality_flags = ["polygon_clipping_edge_present"] if edge_diagnostics["polygon_cut_edge_pixels"] else []
        if full_width_box(region):
            quality_flags.append("full_width_box_non_localizing")
        regions.append({
            **region, "coarse_geometry": region_geometry_kind(region),
            "coarse_polygon_px": points, "coarse_polygon_pixels": int(polygon.sum()),
            "status": status, "reason": reason,
            "raw_ids": [instance["instance_id"] for instance in selected],
            "pass_ids": [instance["instance_id"] for instance in passed],
            "passed_union_pixels": int(union.sum()), "candidate_pixels": int(candidate.sum()),
            "passed_union_overlap_fraction": _fraction(int(candidate.sum()), int(union.sum())),
            "coarse_polygon_support_fraction": _fraction(int(candidate.sum()), int(polygon.sum())),
            "raw_union_polygon_overlap_fraction": _fraction(int(np.count_nonzero(raw_union & polygon)), int(raw_union.sum())),
            "candidate_functional_mask_key": key,
            "candidate_mask_contours": mask_contours(candidate) if key else [],
            "coarse_polygon_preserved": True, "segmentation_fallback_used": False,
            "quality_flags": quality_flags,
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
        "instances": [{k: v for k, v in instance.items() if k != "mask"} for instance in instances],
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
        "boundary_geometry_diagnostics": _boundary_diagnostics(scene["boundaries"], candidates, width, height),
        "boundary_diagnostic_policy": "Only simple open pixel polylines and single convex hole-free candidate contours; otherwise omitted",
        "boundary_support_diagnostics": _boundary_support_diagnostics(scene["boundaries"], passed_unions, width, height),
        "candidate_edge_diagnostic_policy": (
            "Foreground-side 3x3 erosion edges, image frame excluded and counted separately. "
            "An edge pixel with an all-true 3x3 neighborhood in the box-passed raw union is polygon_cut; "
            "otherwise it is a SAM candidate edge, not a verified physical boundary."),
        "quality_flag_policy": (
            "Geometry and unresolved coarse/candidate overlap warnings only; flag presence or absence "
            "does not establish true or false functional semantics. full_width_box_non_localizing records "
            "that the region's VLM box spans the entire 0..1000 width and therefore localizes no horizontal "
            "extent; it is an audit observation about localization, never a role or surface judgment. "
            "Distance support is not boundary verification."),
        "boundary_refinement_performed": False,
        "geometry": {"mode": "image_only", "distance_units": "pixels"},
        "display": {"default": "refined_overlay.jpg", "role_palette_rgb": ROLE_PALETTE,
                    "overlap_style": "role_color_blend", "text_in_image": False,
                    "raw_evidence_path": "raw_evidence.json", "raw_masks_path": "raw_masks.npz"},
        "warnings": [EXTENT_CAVEAT,
                     "Functional roles, relations and boundary polylines remain VLM hypotheses, not verified semantics.",
                     "No detection or rejected support does not establish absence; coarse polygons remain unrefined.",
                     "Overlapping functions are retained independently, not resolved by draw order.",
                     "All prompts share one frame evidence group; geometry diagnostics do not refine boundaries."],
        **GATES,
    }
    return evidence, raw_masks, functional_masks


def _render_regions(image, regions, masks):
    overlay = image.copy()
    counts = np.zeros(image.shape[:2], np.uint8)
    color_sum = np.zeros((*image.shape[:2], 3), np.float64)
    for region in sorted(regions, key=lambda value: value["region_id"]):
        mask = masks.get(region["region_id"])
        if mask is None:
            continue
        counts += mask
        color_sum[mask] += np.asarray(ROLE_PALETTE[region["role"]][::-1], dtype=np.uint8)
    for region in sorted(regions, key=lambda value: value["region_id"]):
        mask = masks.get(region["region_id"])
        if mask is None:
            continue
        color = np.asarray(ROLE_PALETTE[region["role"]][::-1], dtype=np.uint8)
        visible = mask & (counts == 1)
        overlay[visible] = (0.5 * image[visible] + 0.5 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        edges = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(edges, contours, -1, 1, 1)
        overlay[(edges != 0) & (counts <= 1)] = color
    multi = counts > 1
    mean_color = color_sum[multi] / counts[multi, None]
    overlay[multi] = (0.5 * image[multi] + 0.5 * mean_color).astype(np.uint8)
    return overlay


def render_overlays(image, evidence, functional_masks):
    regions = evidence["regions"]
    coarse = {r["region_id"]: _polygon_mask(r["coarse_polygon_px"], image.shape[:2]) for r in regions}
    candidates = {r["region_id"]: functional_masks[r["candidate_functional_mask_key"]]
                  for r in regions if r["candidate_functional_mask_key"] is not None}
    boundary_view = image.copy()
    by_id = {r["region_id"]: r for r in regions}
    for boundary in evidence["vlm_boundaries"]:
        side = by_id.get(boundary["side_a"], by_id.get(boundary["side_b"], {"role": "unknown"}))
        points = np.asarray(pixel_points(boundary["points"], image.shape[1], image.shape[0]), np.int32)
        cv2.polylines(boundary_view, [points], False, ROLE_PALETTE[side["role"]][::-1], 2, cv2.LINE_AA)
    return {"coarse_overlay.jpg": _render_regions(image, regions, coarse),
            "refined_overlay.jpg": _render_regions(image, regions, candidates),
            "boundaries_overlay.jpg": boundary_view}


def run(args):
    # Protect programmatic callers too, not only the CLI entry point.
    with offline_runtime():
        return _run_offline(args)


def _run_offline(args):
    output = Path(args.output).resolve()
    results, scenes = Path(args.results).resolve(), Path(args.scenes).resolve()
    if output.exists() or output.is_relative_to(results) or output.is_relative_to(scenes):
        raise ValueError("Output must be a new directory outside the source and scene runs")
    if not 0 < args.confidence < 1 or type(args.threads) is not int or args.threads < 1:
        raise ValueError("confidence must be in (0,1) and threads must be positive")
    scene_manifest_hash = file_hash(scenes / "manifest.json")
    scene_manifest, prepared = load_scene_run(results, scenes)
    if file_hash(scenes / "manifest.json") != scene_manifest_hash:
        raise ValueError("Scene manifest changed during input preparation")
    core = load_core()
    weights = Path(args.weights).resolve(strict=True)
    configuration = {
        "pipeline_version": PIPELINE_VERSION, "weights_sha256": file_hash(weights),
        "weights_path": str(weights), "clip_dir": str(Path(args.clip_dir).resolve()) if args.clip_dir else None,
        "confidence": args.confidence, "device": args.device, "threads": args.threads,
        "source_manifest_sha256": scene_manifest["configuration"]["source_manifest_sha256"],
        "source_scene_manifest_sha256": scene_manifest_hash,
        "source_scene_record_sha256": {item["frame"]["frame_id"]: item["scene_record_sha256"] for item in prepared},
        "surface_prompts": REGION_PROMPTS, "max_outside_fraction": MAX_OUTSIDE_FRACTION,
        "implementation_sha256": {path.name: file_hash(path) for path in (
            Path(__file__), PKG / "scene" / "scene_understanding.py",
            PKG / "scene" / "run_scene_understanding.py", PKG / "proposals" / "run_proposals.py",
            PKG / "perception" / "sam3_local.py", PKG / "semantic_pipeline.py",
            PKG / "proposals" / "run_offline.py", PKG / "evidence" / "review_results.py",
            Path(core.__file__))},
    }
    if scene_manifest["schema_version"] == "field_functional_scene_staged_run_v1":
        configuration["implementation_sha256"].update(
            {path.name: file_hash(path) for path in
             (PKG / "scene" / "scene_staging.py", PKG / "scene" / "run_staged_scene.py")})
    output.mkdir(parents=True)
    manifest = {"schema_version": RUN_SCHEMA, "configuration": configuration, "status": "running",
                "purpose": "development_functional_candidates_not_held_out_accuracy",
                "requested_frame_count": len(prepared), "network_inference_performed": False,
                "extent_caveat": EXTENT_CAVEAT, **GATES, "frames": []}
    write_json(output / "manifest.json", manifest)
    model = None
    try:
        for item in prepared:
            frame, image, scene = item["frame"], item["image"], item["scene"]
            frame_id = frame["frame_id"]
            queries = scene_queries(scene, image.shape[1], image.shape[0])
            detections, timing = [], {"status": "skipped_no_surface_queries", "total_seconds": 0.0}
            if queries:
                if model is None:
                    model = LocalSAM3(weights, args.clip_dir, args.confidence, args.device, args.threads)
                detections, timing = model.predict_regions(image, queries)
            evidence, raw_masks, functional_masks = refine_regions(image, scene, detections, core, frame_id)
            model_metadata = {
                **(model.metadata if model is not None else {}), "backend": BACKEND,
                "prompt_version": PIPELINE_VERSION, "prompts": queries,
                "vlm_used": False, "vlm_scene_hypotheses_used": True, "inference_performed": bool(queries),
                "weights_sha256": configuration["weights_sha256"], "weights_name": weights.name,
                "confidence_threshold": args.confidence, "device": args.device, "threads": args.threads,
            }
            provenance = {
                "source": {key: frame[key] for key in ("video_name", "video_sha256", "frame_index", "capture_time_s")},
                "input_image_sha256": item["image_sha256"],
                "source_manifest_sha256": configuration["source_manifest_sha256"],
                "source_scene_manifest_sha256": scene_manifest_hash,
                "source_scene_record_sha256": item["scene_record_sha256"],
                "source_scene_record_path": str(scenes / (frame_id + ".json")),
                "model": model_metadata, "timing": timing,
            }
            evidence.update(provenance)
            evidence["raw_evidence"].update(provenance)
            folder = output / frame_id
            folder.mkdir()
            original_name = "frame" + Path(frame["image_path"]).suffix.lower()
            (folder / original_name).write_bytes(item["image_bytes"])
            for name, overlay in render_overlays(image, evidence, functional_masks).items():
                write_image(folder / name, overlay)
            np.savez_compressed(folder / "raw_masks.npz", **raw_masks)
            np.savez_compressed(folder / "functional_masks.npz", **functional_masks)
            write_json(folder / "raw_evidence.json", evidence["raw_evidence"])
            write_json(folder / "scene_evidence.json", evidence)
            exported = {key: frame[key] for key in ("frame_id", "video_name", "video_sha256", "frame_index", "capture_time_s")}
            paths = {"image_path": original_name, "evidence_path": "scene_evidence.json",
                     "raw_evidence_path": "raw_evidence.json", "raw_masks_path": "raw_masks.npz",
                     "functional_masks_path": "functional_masks.npz", "coarse_overlay_path": "coarse_overlay.jpg",
                     "refined_overlay_path": "refined_overlay.jpg", "boundaries_overlay_path": "boundaries_overlay.jpg"}
            exported.update({key: f"{frame_id}/{name}" for key, name in paths.items()})
            exported.update(source_scene_manifest_sha256=scene_manifest_hash,
                            source_scene_record_sha256=item["scene_record_sha256"],
                            input_image_sha256=item["image_sha256"], **GATES)
            exported["artifact_sha256"] = {exported[key]: file_hash(output / exported[key]) for key in paths}
            manifest["frames"].append(exported)
            write_json(output / "manifest.json", manifest)
            print(f"saved {frame_id}; {len(raw_masks)} raw masks; {len(functional_masks)} candidate functional masks", flush=True)
    except Exception as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__)
        write_json(output / "manifest.json", manifest)
        raise
    manifest["status"] = "completed"
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Offline SAM3 grounding of archived VLM functional scene hypotheses")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True, help="Completed functional scene run")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside both input runs")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
