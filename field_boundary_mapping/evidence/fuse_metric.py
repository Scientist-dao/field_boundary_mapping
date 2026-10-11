import argparse
import json
import math
from pathlib import Path

import numpy as np

from field_boundary_mapping.proposals.run_offline import write_json
from field_boundary_mapping.semantic_pipeline import load_core


def confined_file(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Geometry/result paths must stay inside their containing directory")
    return path


def fuse(results, geometry_path, stride=16):
    if stride < 1:
        raise ValueError("stride must be positive")
    core = load_core()
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("region_proposals_used"):
        raise ValueError("Unverified region proposals cannot be fused into a metric ground map")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry["schema_version"] != "field_boundary_geometry_v1":
        raise ValueError("Unsupported geometry schema")
    if geometry["coordinate_convention"] != "ENU_m" or geometry["time_basis"] != "video_relative_seconds":
        raise ValueError("Explicit ENU meters and matching video-relative capture times are required")
    origin = geometry["origin_lat_lon_alt"]
    if len(origin) != 3 or not all(math.isfinite(x) for x in origin):
        raise ValueError("A finite measured world origin is required")
    if not -90 <= origin[0] <= 90 or not -180 <= origin[1] <= 180:
        raise ValueError("World origin latitude/longitude are outside range")
    frames = {f["frame_id"]: f for f in geometry["frames"]}
    if len(frames) != len(geometry["frames"]):
        raise ValueError("Duplicate geometry frame_id")
    if len({f["video_name"] for f in manifest["frames"]}) > 1:
        raise ValueError("Video-relative clocks cannot merge different clips; supply a single-clip result run")
    evidence_map = core.EvidenceMap(geometry["map_id"], geometry["epoch"])
    reports = []
    latest = 0.0
    for item in sorted(manifest["frames"], key=lambda f: f["capture_time_s"]):
        timestamp = item["capture_time_s"]
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid video capture timestamp")
        latest = max(latest, timestamp)
        frame_id = item["frame_id"]
        if frame_id not in frames:
            reports.append({"frame_id": frame_id, "accepted": False, "reason": "missing_synchronized_geometry"})
            continue
        record = frames[frame_id]
        if record["video_sha256"] != item["video_sha256"] or record["frame_index"] != item["frame_index"]:
            raise ValueError("Geometry belongs to a different source video or frame")
        if record["epoch"] != geometry["epoch"]:
            raise ValueError("Geometry epoch differs from the persistent map")
        if record["image_undistorted"] is not True or record["depth_kind"] != "optical_z_m":
            raise ValueError("Only aligned metric optical-axis depth on undistorted image coordinates is supported")
        if record["depth_source"] not in ("stereo", "lidar", "rgbd"):
            raise ValueError("Unscaled monocular or synthetic depth is not a measured-depth source")
        timestamp = item["capture_time_s"]
        for key in ("image_time_s", "pose_time_s", "depth_time_s"):
            if not math.isfinite(record[key]) or abs(record[key] - timestamp) > 0.05:
                raise ValueError("Pose, depth and image timestamps must match within 50ms")
        with np.load(confined_file(results, item["masks_path"]), allow_pickle=False) as masks:
            labels = masks["surface_labels"]
        depth = np.load(confined_file(geometry_path.parent, record["depth_path"]), allow_pickle=False)
        evidence = json.loads(confined_file(results, item["evidence_path"]).read_text(encoding="utf-8"))
        width, height = evidence["image"]["width"], evidence["image"]["height"]
        if depth.shape != (height, width) or labels.shape != depth.shape:
            raise ValueError("Depth, mask, and original image dimensions differ")
        transform = record["world_from_camera"]
        intrinsics = record["intrinsics_fx_fy_cx_cy"]
        # Only the unoccluded ground layer is fused; canopy edges are not ground boundaries.
        points = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                if labels[y, x] != 2:
                    continue
                point = core.project_depth([x, y], float(depth[y, x]), intrinsics, transform, [width, height])
                if point is not None:
                    points.append(point[:2])
        if not points:
            reports.append({"frame_id": frame_id, "accepted": False, "reason": "no_valid_measured_ground_points"})
            continue
        ground_scores = [r["score"] for r in evidence["regions"] if r["semantic"] == "ground"]
        result = evidence_map.update({
            "frame_id": frame_id, "evidence_group": frame_id,
            "capture_time": timestamp, "publish_time": timestamp,
            "epoch": geometry["epoch"], "points_xy_m": points,
            "semantic": "observed_ground", "semantic_score": max(ground_scores),
            "position_sigma_m": record["xy_position_sigma_m"],
            "camera_xy_m": [transform[3], transform[7]], "geometry_source": "measured_depth",
        })
        reports.append({"frame_id": frame_id, **result})
    snapshot = evidence_map.snapshot(latest)
    return {
        "schema_version": "field_boundary_metric_replay_v1", "map": snapshot,
        "origin_lat_lon_alt": origin, "frame_reports": reports,
        "time_semantics": "offline_capture_time_replay_not_current_driving_clearance",
        "boundary_segments": [], "maneuver_verified": False,
        "limitations": [
            "Only visible measured ground samples are mapped; no automatic physical field boundary reconstruction",
            "Input xy_position_sigma_m must include depth, calibration and pose uncertainty",
            "No authorized-area, soil-bearing, slope or swept-vehicle-path verification",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Fuse existing predictions using explicitly supplied synchronized measured depth")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output file; existing metric maps are not overwritten")
    result = fuse(args.results.resolve(strict=True), args.geometry.resolve(strict=True), args.stride)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
