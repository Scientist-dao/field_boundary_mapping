"""Extract auditable single-frame image interfaces from existing proposal masks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import cv2
import numpy as np

from semantic_pipeline import interface_polylines

SCHEMA = "visible_boundaries_v1"
PRIMARY = ("crop", "ground", "woody_vegetation", "water", "ditch")
SURFACE_CLASS = {**{name: name for name in PRIMARY}, "harvested_ground": "ground"}
OCCLUDERS = ("vehicle", "person", "tarp", "pole")
REASONS = {0: "unknown", -1: "conflict", -2: "occlusion", -3: "fov"}
FLAGS = {"semantic_verified": False, "physical_field_boundary_verified": False,
         "maneuver_verified": False, "annotation_used_for_inference": False}
TYPES = ("crop_ground_interface", "vegetation_coverage_interface",
         "ditch_visible_interface", "water_visible_interface")
# Gap closure never joins woody canopy edges: a canopy silhouette legitimately
# floats over unobserved ground, so an unknown band beside trees is not evidence
# of a missing surface-to-surface seam.
GAP_CLOSURE_CLASSES = ("crop", "ground", "water", "ditch")
GAP_TYPES = ("crop_ground_interface", "water_visible_interface", "ditch_visible_interface")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained_file(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Invalid artifact path")
    rel = Path(relative)
    if rel.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError("Invalid artifact path")
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Artifact escapes source run")
    return path


def source_manifest(run_dir):
    root = Path(run_dir).resolve(strict=True)
    path = contained_file(root, "manifest.json")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "completed" or not manifest.get("frames"):
        raise ValueError("A completed nonempty source run is required")
    frames = manifest["frames"]
    ids = [f["frame_id"] for f in frames]
    if len(set(ids)) != len(ids) or any(not isinstance(fid, str) or not fid or
                                      any(c in fid for c in "/\\:") or fid in (".", "..") for fid in ids):
        raise ValueError("Invalid or duplicate frame identity")
    if len({(f["video_name"], f["video_sha256"]) for f in frames}) != 1:
        raise ValueError("Boundary run must refer to one source video")
    times = [f["capture_time_s"] for f in frames]
    if any(not isinstance(t, (float, int)) or isinstance(t, bool) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("Invalid capture timestamp")
    if len(set(times)) != len(times) or len({f["frame_index"] for f in frames}) != len(frames):
        raise ValueError("Duplicate source frame")
    return root, manifest, hashlib.sha256(raw).hexdigest()


def source_artifacts(root, frame):
    paths, hashes = {}, {}
    for field in ("image_path", "evidence_path", "masks_path"):
        rel = frame[field]
        path = contained_file(root, rel)
        expected = frame.get("artifact_sha256", {}).get(rel)
        if not expected or sha256(path) != expected:
            raise ValueError("Source artifact hash mismatch")
        paths[field] = path
        hashes[rel] = expected
    return paths, hashes


def compact_line(line):
    if len(line) < 3:
        return line
    points = np.asarray(line)
    directions = np.diff(points, axis=0)
    turns = np.any(directions[1:] != directions[:-1], axis=1)
    return points[np.r_[True, turns, True]].tolist()


def line_neighbors(line, shape):
    points = np.asarray(line)
    mids = (points[1:] + points[:-1]) / 2
    vertical = points[1:, 0] == points[:-1, 0]
    x0 = np.floor(mids[:, 0]).astype(int)
    y0 = np.floor(mids[:, 1]).astype(int)
    x1 = x0 + vertical.astype(int)
    y1 = y0 + (~vertical).astype(int)
    xx, yy = np.r_[x0, x1], np.r_[y0, y1]
    inside = (xx >= 0) & (xx < shape[1]) & (yy >= 0) & (yy < shape[0])
    return yy[inside], xx[inside]


def endpoint_reasons(point, labels):
    x, y = point
    h, w = labels.shape
    values = set()
    for py in (math.floor(y), math.ceil(y)):
        for px in (math.floor(x), math.ceil(x)):
            values.add(int(labels[py, px]) if 0 <= px < w and 0 <= py < h else -3)
    reasons = [reason for value, reason in REASONS.items() if value in values]
    four = [int(labels[py, px]) if 0 <= px < w and 0 <= py < h else -3
            for px, py in ((math.floor(x), math.floor(y)), (math.ceil(x), math.floor(y)),
                           (math.ceil(x), math.ceil(y)), (math.floor(x), math.ceil(y))) ]
    crossing = all(value > 0 for value in four) and all(four[i] != four[(i + 1) % 4] for i in range(4))
    if len(values - set(REASONS)) > 2 or crossing:
        reasons.append("junction")
    if not reasons:
        reasons.append("interface_end")
    return reasons


def boundary_type(a, b):
    if "ditch" in (a, b):
        return "ditch_visible_interface"
    if "water" in (a, b):
        return "water_visible_interface"
    if "woody_vegetation" in (a, b):
        return "vegetation_coverage_interface"
    return "crop_ground_interface"


def close_surface_gaps(labels, gap_closure_px):
    """Assign narrow unknown bands between two eligible surface classes.

    An unknown pixel joins its nearest eligible surface class when the sum of
    its distances to that class and to the nearest *different* eligible class
    is at most gap_closure_px + 2; a uniform k-pixel band therefore closes
    when gap_closure_px >= k - 1. Conflict, occlusion and FOV pixels are never
    reassigned. Distances are 5x5-mask approximate Euclidean and are not
    occlusion-aware: bands may close around an occluder, but the occluder
    itself is never relabelled.
    """
    shape = labels.shape
    eligible = [index for index, name in enumerate(PRIMARY, 1) if name in GAP_CLOSURE_CLASSES]
    present = [index for index in eligible if bool((labels == index).any())]
    empty = {"assigned": np.zeros(shape, bool), "present": [], "distances": {}, "nearest": {},
             "origins": {}, "stats": {"assigned_pixels": 0, "by_pair": {}, "by_class": {}}}
    if len(present) < 2:
        return labels.copy(), empty
    distances, nearest, origins = {}, {}, {}
    for index in present:
        source = (labels != index).astype(np.uint8)
        dist, lab = cv2.distanceTransformWithLabels(
            source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        # DIST_LABEL_PIXEL numbers the zero pixels 1..K in row-major scan order,
        # so label k decodes through the k-th nonzero coordinate, not divmod(k, w).
        ys, xs = np.nonzero(labels == index)
        distances[index] = dist
        nearest[index] = lab
        origins[index] = (ys, xs)
    stacked = np.stack([distances[index] for index in present])
    ranking = np.argsort(stacked, axis=0)
    first = ranking[0]
    second = ranking[1]
    first_index = np.asarray(present)[first]
    second_index = np.asarray(present)[second]
    nearest_first = np.take_along_axis(stacked, first[None], axis=0)[0]
    nearest_second = np.take_along_axis(stacked, second[None], axis=0)[0]
    assigned = (labels == 0) & (nearest_first + nearest_second <= gap_closure_px + 2)
    closed = labels.copy()
    closed[assigned] = first_index[assigned]
    by_pair, by_class = {}, {}
    for pair in ({frozenset(map(int, values)) for values in
                  zip(first_index[assigned].tolist(), second_index[assigned].tolist())}):
        names = sorted(PRIMARY[index - 1] for index in pair)
        key = "|".join(names)
        mask = assigned & np.isin(first_index, list(pair)) & np.isin(second_index, list(pair))
        by_pair[key] = int(mask.sum())
    for index in present:
        count = int((assigned & (first_index == index)).sum())
        if count:
            by_class[PRIMARY[index - 1]] = count
    stats = {"assigned_pixels": int(assigned.sum()), "by_pair": by_pair, "by_class": by_class}
    return closed, {"assigned": assigned, "present": present, "distances": distances,
                    "nearest": nearest, "origins": origins, "stats": stats}


def extract_frame(evidence, masks, min_length_px=12, gap_closure_px=0):
    if not math.isfinite(min_length_px) or min_length_px < 0:
        raise ValueError("min_length_px must be finite and nonnegative")
    if not math.isfinite(gap_closure_px) or gap_closure_px < 0:
        raise ValueError("gap_closure_px must be finite and nonnegative")
    h, w = evidence["image"]["height"], evidence["image"]["width"]
    if type(h) is not int or type(w) is not int or min(h, w) < 1:
        raise ValueError("Invalid image dimensions")
    shape = (h, w)
    unions = {name: np.zeros(shape, bool) for name in PRIMARY}
    blocked = np.zeros(shape, bool)
    members = {name: [] for name in PRIMARY}
    used, rejected, unsupported = [], [], []
    identities = set()
    for instance in evidence["instances"]:
        iid = instance["instance_id"]
        if iid in identities:
            raise ValueError("Duplicate instance identity")
        identities.add(iid)
        mask = np.asarray(masks[instance["mask_key"]])
        if mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError("Mask must be boolean and match source image dimensions")
        if instance["spatial_check"]["status"] != "spatially_consistent":
            rejected.append(iid)
            continue
        semantic = instance["semantic"]
        if semantic in SURFACE_CLASS:
            primary = SURFACE_CLASS[semantic]
            unions[primary] |= mask
            members[primary].append((instance, mask))
        elif semantic in OCCLUDERS:
            blocked |= mask
        else:
            unsupported.append(iid)
            continue
        used.append(iid)
    counts = np.stack(list(unions.values())).sum(axis=0)
    labels = np.zeros(shape, np.int8)
    for index, name in enumerate(PRIMARY, 1):
        labels[unions[name] & (counts == 1)] = index
    labels[counts > 1] = -1
    labels[blocked] = -2
    if gap_closure_px > 0:
        closed_labels, closure = close_surface_gaps(labels, gap_closure_px)
    else:
        closed_labels, closure = labels, None
    boundaries, contact_audit = [], []
    for (aid, a), (bid, b) in combinations(enumerate(PRIMARY, 1), 2):
        lines = interface_polylines(labels, aid, bid)
        if not lines:
            continue
        audit = {"side_a": a, "side_b": b, "raw_segments": len(lines),
                 "raw_length_px": sum(len(line) - 1 for line in lines),
                 "short_segments": 0, "short_length_px": 0}
        for line in lines:
            length = len(line) - 1
            if length < min_length_px:
                audit["short_segments"] += 1
                audit["short_length_px"] += length
                continue
            yy, xx = line_neighbors(line, shape)
            sides = []
            for name, label in ((a, aid), (b, bid)):
                side_pixels = labels[yy, xx] == label
                supporting = [i for i, mask in members[name] if np.any(mask[yy, xx] & side_pixels)]
                sides.append({"semantic": name,
                              "instance_ids": sorted(i["instance_id"] for i in supporting),
                              "source_semantics": sorted({i["semantic"] for i in supporting})})
            closed = line[0] == line[-1]
            boundaries.append({
                "boundary_id": f"B{len(boundaries) + 1:03d}",
                "type": boundary_type(a, b), "side_a": sides[0], "side_b": sides[1],
                "points_px": compact_line(line), "length_px": float(length), "closed": closed,
                "endpoint_reasons": [["closed"], ["closed"]] if closed else
                                    [endpoint_reasons(line[0], labels), endpoint_reasons(line[-1], labels)],
                "status": "image_candidate", "evidence_frame_id": evidence["frame_id"], **FLAGS,
            })
        contact_audit.append(audit)
    diagnostics, coverage = [], []
    short_coverage_count, short_coverage_length = 0, 0
    padded = np.pad(labels, 1, constant_values=-3)
    for label, name in enumerate(PRIMARY, 1):
        for value, reason in REASONS.items():
            lines = interface_polylines(padded if value == -3 else labels, label, value)
            if not lines:
                continue
            if value == 0 and name != "ground":
                for line in lines:
                    length = len(line) - 1
                    if length < min_length_px:
                        short_coverage_count += 1
                        short_coverage_length += length
                        continue
                    yy, xx = line_neighbors(line, shape)
                    on_side = labels[yy, xx] == label
                    supporting = [i for i, mask in members[name] if np.any(mask[yy, xx] & on_side)]
                    closed = line[0] == line[-1]
                    coverage.append({
                        "boundary_id": f"C{len(coverage) + 1:03d}", "type": name + "_coverage_contour",
                        "side_a": {"semantic": name, "instance_ids": sorted(i["instance_id"] for i in supporting),
                                   "source_semantics": sorted({i["semantic"] for i in supporting})},
                        "side_b": {"semantic": "unknown", "instance_ids": [], "source_semantics": []},
                        "points_px": compact_line(line), "length_px": float(length), "closed": closed,
                        "endpoint_reasons": [["closed"], ["closed"]] if closed else
                                            [endpoint_reasons(line[0], labels), endpoint_reasons(line[-1], labels)],
                        "status": "one_sided_candidate", "evidence_frame_id": evidence["frame_id"], **FLAGS,
                    })
            length = sum(len(line) - 1 for line in lines)
            compact = [compact_line(line) for line in lines]
            if value == -3:
                compact = [(np.asarray(line) - 1).tolist() for line in compact]
            diagnostics.append({"reason": reason, "semantic": name,
                                "polylines_px": compact, "length_px": float(length)})
    gap_closed, short_gap_closed_segments, short_gap_closed_length = [], 0, 0
    if closure is not None and closure["assigned"].any():
        eligible = [(index, name) for index, name in enumerate(PRIMARY, 1) if name in GAP_CLOSURE_CLASSES]
        present = set(closure["present"])
        for (aid, a), (bid, b) in combinations([pair for pair in eligible if pair[0] in present], 2):
            for line in interface_polylines(closed_labels, aid, bid):
                yy, xx = line_neighbors(line, shape)
                band = closure["assigned"][yy, xx]
                if not band.any():
                    continue
                length = len(line) - 1
                if length < min_length_px:
                    short_gap_closed_segments += 1
                    short_gap_closed_length += length
                    continue
                sides, assigned_sides = [], []
                for name, label in ((a, aid), (b, bid)):
                    side_pixels = closed_labels[yy, xx] == label
                    ys, xs = closure["origins"][label]
                    k = closure["nearest"][label][yy, xx].astype(np.int64)
                    ny, nx = ys[k - 1], xs[k - 1]
                    supporting = [i for i, mask in members[name] if np.any(mask[ny, nx] & side_pixels)]
                    sides.append({"semantic": name,
                                  "instance_ids": sorted(i["instance_id"] for i in supporting),
                                  "source_semantics": sorted({i["semantic"] for i in supporting})})
                    assigned_sides.append(int((band & side_pixels).sum()))
                widths = closure["distances"][aid][yy, xx][band] + closure["distances"][bid][yy, xx][band]
                closed = line[0] == line[-1]
                gap_closed.append({
                    "boundary_id": f"GC{len(gap_closed) + 1:03d}",
                    "type": boundary_type(a, b), "side_a": sides[0], "side_b": sides[1],
                    "points_px": compact_line(line), "length_px": float(length), "closed": closed,
                    "endpoint_reasons": [["closed"], ["closed"]] if closed else
                                        [endpoint_reasons(line[0], closed_labels), endpoint_reasons(line[-1], closed_labels)],
                    "status": "gap_closed_candidate", "evidence_frame_id": evidence["frame_id"],
                    "gap": {"assigned_neighbor_px": int(band.sum()),
                            "side_a_assigned_px": assigned_sides[0], "side_b_assigned_px": assigned_sides[1],
                            "width_px": {"min": float(widths.min()), "mean": float(widths.mean()),
                                         "max": float(widths.max())}}, **FLAGS,
                })
    histogram = {name: int((labels == index).sum()) for index, name in enumerate(PRIMARY, 1)}
    histogram.update({reason: int((labels == value).sum()) for value, reason in REASONS.items() if value != -3})
    closure_stats = dict(closure["stats"]) if closure is not None else {"assigned_pixels": 0, "by_pair": {}, "by_class": {}}
    closure_stats.update({"gap_closed_count": len(gap_closed),
                          "gap_closed_length_px": sum(b["length_px"] for b in gap_closed),
                          "short_gap_closed_segments": short_gap_closed_segments,
                          "short_gap_closed_length_px": short_gap_closed_length})
    result = {
        "frame_id": evidence["frame_id"], "image": {"width": w, "height": h},
        "boundaries": boundaries, "coverage_contours": coverage, "diagnostics": diagnostics,
        "gap_closed_boundaries": gap_closed,
        "status": "image_candidates" if boundaries else "no_supported_interface",
        "stats": {"used_instance_ids": sorted(used), "rejected_instance_ids": sorted(rejected),
                  "unsupported_instance_ids": sorted(unsupported), "label_pixels": histogram,
                  "contact_audit": contact_audit, "boundary_count": len(boundaries),
                  "boundary_length_px": sum(b["length_px"] for b in boundaries),
                  "short_segments": sum(a["short_segments"] for a in contact_audit),
                  "short_length_px": sum(a["short_length_px"] for a in contact_audit),
                  "coverage_count": len(coverage), "coverage_length_px": sum(c["length_px"] for c in coverage),
                  "short_coverage_count": short_coverage_count, "short_coverage_length_px": short_coverage_length,
                  "gap_closure": closure_stats}, **FLAGS,
    }
    return result, labels


def extract_run(run_dir, min_length_px=12, gap_closure_px=0):
    root, manifest, manifest_hash = source_manifest(run_dir)
    frames = []
    for entry in sorted(manifest["frames"], key=lambda f: f["capture_time_s"]):
        paths, hashes = source_artifacts(root, entry)
        evidence = json.loads(paths["evidence_path"].read_bytes())
        image = cv2.imdecode(np.frombuffer(paths["image_path"].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or evidence["image"] != {"width": image.shape[1], "height": image.shape[0]}:
            raise ValueError("Source image dimensions do not match evidence")
        if evidence["frame_id"] != entry["frame_id"] or any(
            evidence.get("source", {}).get(key) != entry[key]
            for key in ("video_name", "video_sha256", "frame_index", "capture_time_s")
        ):
            raise ValueError("Frame evidence identity mismatch")
        with np.load(paths["masks_path"], allow_pickle=False) as masks:
            frame, _ = extract_frame(evidence, masks, min_length_px, gap_closure_px)
        frame["capture_time_s"] = entry["capture_time_s"]
        frame["source"] = {key: entry[key] for key in ("video_name", "video_sha256", "frame_index")}
        frame["source"]["artifact_sha256"] = hashes
        frames.append(frame)
    if sha256(root / "manifest.json") != manifest_hash:
        raise ValueError("Source manifest changed during extraction")
    return {
        "schema_version": SCHEMA, "run": root.name, "source_manifest_sha256": manifest_hash,
        "parameters": {"min_length_px": min_length_px, "connectivity": 4,
                       "gap_bridging_px": 0, "gap_closure_px": gap_closure_px,
                       "geometry_simplification": "collinear_only",
                       "surface_class": SURFACE_CLASS, "occluders": list(OCCLUDERS),
                       "instance_policy": "union_of_spatially_consistent_masks",
                       "temporal_identity_used": False},
        "implementation_sha256": {p.name: sha256(p) for p in
                                  (Path(__file__), Path(__file__).with_name("semantic_pipeline.py"))},
        "frames": frames,
        "summary": {"frames": len(frames), "frames_with_boundaries": sum(bool(f["boundaries"]) for f in frames),
                    "boundary_count": sum(len(f["boundaries"]) for f in frames),
                    "boundary_length_px": sum(f["stats"]["boundary_length_px"] for f in frames),
                    "coverage_count": sum(len(f["coverage_contours"]) for f in frames),
                    "frames_with_coverage": sum(bool(f["coverage_contours"]) for f in frames),
                    "gap_closed_count": sum(len(f["gap_closed_boundaries"]) for f in frames),
                    "frames_with_gap_closed": sum(bool(f["gap_closed_boundaries"]) for f in frames),
                    "gap_closed_length_px": sum(b["length_px"] for f in frames
                                                for b in f["gap_closed_boundaries"]),
                    "types": dict(Counter(b["type"] for f in frames for b in f["boundaries"]))},
        "notes": ["Single-frame mask contact, not a measured physical field boundary or a maneuver permit.",
                  "Crop/woody coverage edges may occlude the actual ground edge; ditch near/far side is unknown.",
                  "Ground is a coarse surface class including ground and harvested_ground proposals; subtypes are not adjudicated.",
                  "Only spatially consistent masks support interfaces; rejected masks are retained in the source run, not evidence of absence.",
                  "Unknown gaps, incompatible overlaps, objects and FOV edges are never bridged; short contacts are counted separately.",
                  "Gap-closed candidates reassign narrow unknown bands to the nearest eligible surface class (never woody vegetation, conflict, occlusion or FOV); they are hypothetical geometry, not observed interfaces, and never upgrade B candidates.",
                  "B IDs are local to each frame, not tracked physical identities; capture_time_s is the actual sampled timestamp.",
                  "No independent manual boundary reference has been evaluated; extraction counts are not accuracy metrics."], **FLAGS,
    }


def load_boundary_payload(path, run_dir):
    payload = json.loads(Path(path).read_bytes())
    root, manifest, manifest_hash = source_manifest(run_dir)
    if payload.get("schema_version") != SCHEMA or payload.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("Boundary evidence does not match source run")
    if any(payload.get(key) is not False for key in FLAGS):
        raise ValueError("Boundary evidence must remain unverified")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Invalid boundary parameters")
    parameters = dict(parameters)
    parameters.setdefault("gap_closure_px", 0)
    # Payloads written before gap closure existed replay with closure disabled.
    frames = []
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            raise ValueError("Invalid boundary frame")
        frame = dict(frame)
        frame.setdefault("gap_closed_boundaries", [])
        stats = frame.get("stats")
        if not isinstance(stats, dict):
            raise ValueError("Invalid boundary frame stats")
        stats = dict(stats)
        stats.setdefault("gap_closure", {"assigned_pixels": 0, "by_pair": {}, "by_class": {},
                                         "gap_closed_count": 0, "gap_closed_length_px": 0.0,
                                         "short_gap_closed_segments": 0, "short_gap_closed_length_px": 0})
        frame["stats"] = stats
        frames.append(frame)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Invalid boundary summary")
    summary = dict(summary)
    summary.setdefault("gap_closed_count", 0)
    summary.setdefault("frames_with_gap_closed", 0)
    summary.setdefault("gap_closed_length_px", 0.0)
    normalized = {**payload, "parameters": parameters, "frames": frames, "summary": summary}
    expected = {f["frame_id"]: f for f in manifest["frames"]}
    actual = [f["frame_id"] for f in frames]
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError("Boundary frame set does not match source run")
    for frame in frames:
        entry = expected[frame["frame_id"]]
        paths, hashes = source_artifacts(root, entry)
        if frame.get("source", {}).get("artifact_sha256") != hashes or frame["capture_time_s"] != entry["capture_time_s"]:
            raise ValueError("Boundary source binding mismatch")
        if any(frame["source"].get(key) != entry[key] for key in ("video_name", "video_sha256", "frame_index")):
            raise ValueError("Boundary source identity mismatch")
        evidence = json.loads(paths["evidence_path"].read_bytes())
        if frame["image"] != evidence["image"] or any(frame.get(key) is not False for key in FLAGS):
            raise ValueError("Boundary frame dimensions or verification flags invalid")
        candidates = []
        for key, valid_types, status, semantics in (
                ("boundaries", TYPES, "image_candidate", PRIMARY),
                ("coverage_contours", tuple(name + "_coverage_contour" for name in PRIMARY if name != "ground"),
                 "one_sided_candidate", PRIMARY),
                ("gap_closed_boundaries", GAP_TYPES, "gap_closed_candidate", GAP_CLOSURE_CLASSES)):
            for boundary in frame.get(key, []):
                candidates.append((boundary, valid_types, status, semantics, key == "coverage_contours"))
        ids = [b["boundary_id"] for b, *_ in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate boundary identity")
        lines = []
        by_id = {i["instance_id"]: i for i in evidence["instances"]}
        for boundary, valid_types, status, semantics, one_sided in candidates:
            if boundary["type"] not in valid_types or boundary["status"] != status or any(
                    boundary.get(key) is not False for key in FLAGS) or boundary["evidence_frame_id"] != frame["frame_id"]:
                raise ValueError("Invalid boundary candidate")
            if not math.isfinite(boundary["length_px"]) or boundary["length_px"] <= 0:
                raise ValueError("Invalid boundary length")
            sides = (boundary["side_a"],) if one_sided else (boundary["side_a"], boundary["side_b"])
            if one_sided and boundary["side_b"] != {"semantic": "unknown", "instance_ids": [], "source_semantics": []}:
                raise ValueError("One-sided contour cannot claim an observed other side")
            for side in sides:
                if side["semantic"] not in semantics or not side["instance_ids"] or not set(side["instance_ids"]) <= set(by_id):
                    raise ValueError("Invalid boundary support")
                instances = [by_id[iid] for iid in side["instance_ids"]]
                if any(SURFACE_CLASS.get(i["semantic"]) != side["semantic"] or
                       i["spatial_check"]["status"] != "spatially_consistent" for i in instances):
                    raise ValueError("Boundary support must match a spatially consistent semantic")
                if side["source_semantics"] != sorted({i["semantic"] for i in instances}):
                    raise ValueError("Boundary subtype provenance mismatch")
            if not one_sided and boundary["side_a"]["semantic"] == boundary["side_b"]["semantic"]:
                raise ValueError("Two-sided interface requires distinct primary classes")
            points = np.asarray(boundary["points_px"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                raise ValueError("Invalid boundary polyline")
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if not math.isclose(length, boundary["length_px"]):
                raise ValueError("Boundary geometry and length mismatch")
            if boundary["closed"] != bool(np.array_equal(points[0], points[-1])):
                raise ValueError("Boundary closure mismatch")
            lines.append(boundary["points_px"])
        for diagnostic in frame["diagnostics"]:
            if diagnostic["reason"] not in REASONS.values() or diagnostic["semantic"] not in PRIMARY:
                raise ValueError("Invalid boundary diagnostic")
            lines.extend(diagnostic["polylines_px"])
        for line in lines:
            points = np.asarray(line, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError("Invalid boundary polyline")
            if np.any(points < -0.5) or np.any(points[:, 0] > frame["image"]["width"] - 0.5) or np.any(
                    points[:, 1] > frame["image"]["height"] - 0.5):
                raise ValueError("Boundary polyline outside source image")
    minimum = parameters.get("min_length_px")
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0:
        raise ValueError("Invalid boundary extraction threshold")
    gap_closure = parameters["gap_closure_px"]
    if type(gap_closure) not in (int, float) or not math.isfinite(gap_closure) or gap_closure < 0:
        raise ValueError("Invalid boundary gap closure parameter")
    # Source hashes authenticate inputs, not the derived geometry or its claimed support.
    replay = extract_run(root, min_length_px=minimum, gap_closure_px=gap_closure)
    for key in ("run", "parameters", "frames", "summary"):
        if normalized.get(key) != replay[key]:
            raise ValueError(f"Boundary {key} does not match source-mask replay")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读提取同帧可见交界候选，不执行模型、不使用轨线身份")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="新的派生 JSON，禁止覆盖源 run 或已有结果")
    parser.add_argument("--min-length-px", type=float, default=12)
    parser.add_argument("--gap-closure-px", type=float, default=0,
                        help="闭合窄未知带的距离和阈值（像素）；0=关闭，仅观测接触入 B")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(args.run.resolve()) or output.exists():
        parser.error("Output must be a new file outside the source run")
    if not output.parent.is_dir():
        parser.error("Output parent directory must exist")
    payload = extract_run(args.run, args.min_length_px, args.gap_closure_px)
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(data)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
