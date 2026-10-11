from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np


SURFACES = ("crop", "ground", "woody_vegetation", "water")
HAZARDS = ("ditch", "person", "vehicle", "tarp", "pole")
COLORS = {
    "crop": (74, 190, 95), "ground": (197, 151, 92),
    "woody_vegetation": (36, 112, 77), "water": (53, 157, 231),
    "ditch": (240, 110, 38), "person": (242, 65, 75),
    "vehicle": (253, 196, 50), "tarp": (139, 126, 238),
    "pole": (229, 119, 190), "conflict": (220, 65, 220),
}


def load_core():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
    try:
        import field_boundary_core
    except ImportError as exc:
        raise RuntimeError("Build field_boundary_mapping with CMake before running inference") from exc
    return field_boundary_core


def mask_contours(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [{
        "points": c.reshape(-1, 2).tolist(), "hole": bool(hierarchy[0, i, 3] >= 0),
    } for i, c in enumerate(contours) if len(c) >= 3]


def interface_polylines(labels, a, b):
    graph = defaultdict(set)

    def edge(p, q):
        graph[p].add(q)
        graph[q].add(p)

    left, right = labels[:, :-1], labels[:, 1:]
    yy, xx = np.nonzero(((left == a) & (right == b)) | ((left == b) & (right == a)))
    for y, x in zip(yy.tolist(), xx.tolist()):
        edge((2 * x + 1, 2 * y - 1), (2 * x + 1, 2 * y + 1))
    top, bottom = labels[:-1], labels[1:]
    yy, xx = np.nonzero(((top == a) & (bottom == b)) | ((top == b) & (bottom == a)))
    for y, x in zip(yy.tolist(), xx.tolist()):
        edge((2 * x - 1, 2 * y + 1), (2 * x + 1, 2 * y + 1))
    lines = []
    degree = {point: len(neighbors) for point, neighbors in graph.items()}
    starts = sorted(p for p in graph if degree[p] != 2) + sorted(graph)
    for start in starts:
        while graph[start]:
            line = [start]
            current = start
            while graph[current]:
                following = min(graph[current])
                graph[current].remove(following)
                graph[following].remove(current)
                line.append(following)
                current = following
                if current == start or degree[current] != 2:
                    break
            lines.append([[x / 2, y / 2] for x, y in line])
    return lines


def render_overlay(image, masks, conflict, lines):
    overlay = image.copy()
    visible = {}
    for name, mask in masks.items():
        visible[name] = (mask & ~conflict) if name in SURFACES else mask
        color = np.asarray(COLORS[name][::-1], dtype=float)
        overlay[visible[name]] = np.clip(0.45 * overlay[visible[name]] + 0.55 * color, 0, 255).astype(np.uint8)
    overlay[conflict] = np.asarray(COLORS["conflict"][::-1], dtype=np.uint8)
    for name, mask in visible.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    for line in lines:
        points = np.rint(np.asarray(line)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [points], False, (0, 0, 0), 3)
    return overlay


def analyze_frame(image, detections, frame_id, core, row_context=None):
    shape = image.shape[:2]
    masks = {name: np.zeros(shape, dtype=bool) for name in (*SURFACES, *HAZARDS)}
    scores = defaultdict(list)
    for detection in detections:
        name = detection["semantic"]
        mask = np.asarray(detection["mask"])
        score = float(detection["score"])
        if name not in masks or mask.shape != shape or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid model detection")
        masks[name] |= mask.astype(bool)
        scores[name].append(score)
    stack = np.stack([masks[name] for name in SURFACES])
    counts = stack.sum(axis=0)
    conflict = counts > 1
    labels = np.zeros(shape, dtype=np.uint8)
    for index, name in enumerate(SURFACES, 1):
        labels[masks[name] & ~conflict] = index
    blocked = np.zeros(shape, dtype=bool)
    for name in HAZARDS:
        blocked |= masks[name]
    # Hazard/structure masks are a separate layer, never proof of visible ground underneath.
    labels[blocked] = 0
    conflict &= ~blocked
    lines = interface_polylines(labels, 1, 2)
    contact_length = sum(len(line) - 1 for line in lines)
    contact_fraction = min(1.0, contact_length / max(shape[1], 1))
    row_context = row_context or {}
    hypothesis = core.assess_headland(
        float(np.mean(labels == 2)), contact_fraction,
        int(row_context.get("confirmed_row_ends", 0)),
        bool(row_context.get("rows_continue", False)),
        bool(row_context.get("row_end_at_fov_edge", False)),
    )
    regions = []
    hazards = []
    for name, mask in masks.items():
        if not mask.any():
            continue
        region = {
            "semantic": name, "score": max(scores[name]),
            "instance_scores": scores[name], "area_fraction": float(mask.mean()),
            "contours": mask_contours(mask), "evidence_frame_id": frame_id,
            "status": "model_candidate",
        }
        (hazards if name in HAZARDS else regions).append(region)
    evidence = {
        "schema_version": "field_boundary_frame_v1", "frame_id": frame_id,
        "image": {"width": shape[1], "height": shape[0]},
        "stage_prediction": "UNKNOWN",
        "stage_reason": "SAM3 segments observable concepts; temporal navigation stage requires a validated semantic/row model",
        "hypothesis": hypothesis,
        "geometry": {
            "mode": "image_only",
            "missing": ["synchronized_pose", "camera_calibration", "metric_depth"],
        },
        "row_structure": {"available": bool(row_context), "context": row_context},
        "regions": regions, "hazards": hazards,
        "boundaries": [{
            "type": "crop_ground_interface", "polylines_px": lines,
            "status": "image_candidate", "evidence_frame_id": frame_id,
            "physical_field_boundary_verified": False,
        }] if lines else [],
        "unknown_fraction": float(np.mean((counts == 0) | blocked)),
        "conflict_fraction": float(conflict.mean()),
        "warnings": [
            "No VLM or trained navigation-stage classifier is active in this baseline",
            "Image contact is not measured ground contact; crop canopy may occlude the soil edge",
            "No detection does not establish absence of a person, ditch, or other hazard",
            "Unknown authorization; maneuver remains unverified",
        ],
        "world_map": None,
    }
    arrays = {**masks, "surface_labels": labels, "surface_conflict": conflict}
    return evidence, render_overlay(image, masks, conflict, lines), arrays
