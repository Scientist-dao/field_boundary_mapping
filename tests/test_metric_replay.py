import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuse_metric import fuse
from run_offline import write_json


class MetricReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = {
            "frames": [{
                "frame_id": "synthetic-f0", "capture_time_s": 0.5, "video_name": "synthetic.mp4",
                "video_sha256": "a" * 64, "frame_index": 5,
                "masks_path": "masks.npz", "evidence_path": "evidence.json",
            }],
        }
        np.savez_compressed(self.root / "masks.npz", surface_labels=np.full((4, 4), 2, np.uint8))
        np.save(self.root / "depth.npy", np.full((4, 4), 2.0, np.float32))
        write_json(self.root / "evidence.json", {
            "image": {"width": 4, "height": 4}, "regions": [{"semantic": "ground", "score": 0.8}],
        })
        self.geometry = {
            "schema_version": "field_boundary_geometry_v1", "coordinate_convention": "ENU_m",
            "time_basis": "video_relative_seconds", "map_id": "synthetic-map", "epoch": "epoch-1",
            "origin_lat_lon_alt": [30.0, 110.0, 50.0],
            "frames": [{
                "frame_id": "synthetic-f0", "epoch": "epoch-1", "image_undistorted": True,
                "video_sha256": "a" * 64, "frame_index": 5,
                "depth_kind": "optical_z_m", "depth_source": "stereo", "depth_path": "depth.npy",
                "image_time_s": 0.5, "pose_time_s": 0.5, "depth_time_s": 0.5,
                "world_from_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                "intrinsics_fx_fy_cx_cy": [4, 4, 2, 2], "xy_position_sigma_m": 0.1,
            }],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_fusion(self):
        write_json(self.root / "manifest.json", self.manifest)
        write_json(self.root / "geometry.json", self.geometry)
        return fuse(self.root, self.root / "geometry.json", stride=1)

    def test_measured_ground_only_and_no_safety_claim(self):
        result = self.run_fusion()
        self.assertFalse(result["maneuver_verified"])
        self.assertEqual(result["boundary_segments"], [])
        self.assertTrue(result["frame_reports"][0]["accepted"])

    def test_missing_geometry_is_reported(self):
        self.geometry["frames"] = []
        result = self.run_fusion()
        self.assertEqual(result["frame_reports"][0]["reason"], "missing_synchronized_geometry")

    def test_no_ground_does_not_project_canopy(self):
        np.savez_compressed(self.root / "masks.npz", surface_labels=np.ones((4, 4), np.uint8))
        result = self.run_fusion()
        self.assertEqual(result["frame_reports"][0]["reason"], "no_valid_measured_ground_points")

    def test_invalid_depth_stays_unknown(self):
        np.save(self.root / "depth.npy", np.full((4, 4), np.nan))
        result = self.run_fusion()
        self.assertEqual(result["frame_reports"][0]["reason"], "no_valid_measured_ground_points")

    def test_time_mismatch_rejected(self):
        self.geometry["frames"][0]["pose_time_s"] = 0.8
        with self.assertRaises(ValueError):
            self.run_fusion()

    def test_scale_and_image_mismatch_rejected(self):
        for key, value in (("depth_kind", "ray_range_m"), ("depth_source", "monocular"), ("image_undistorted", False)):
            with self.subTest(key=key):
                original = self.geometry["frames"][0][key]
                self.geometry["frames"][0][key] = value
                with self.assertRaises(ValueError):
                    self.run_fusion()
                self.geometry["frames"][0][key] = original

    def test_wrong_depth_resolution_rejected(self):
        np.save(self.root / "depth.npy", np.ones((8, 8)))
        with self.assertRaises(ValueError):
            self.run_fusion()

    def test_epoch_mismatch_rejected(self):
        self.geometry["frames"][0]["epoch"] = "epoch-2"
        with self.assertRaises(ValueError):
            self.run_fusion()

    def test_two_video_clocks_must_not_mix(self):
        self.manifest["frames"].append({**self.manifest["frames"][0], "frame_id": "different", "video_name": "second.mp4"})
        with self.assertRaises(ValueError):
            self.run_fusion()

    def test_depth_path_cannot_escape_geometry_folder(self):
        self.geometry["frames"][0]["depth_path"] = str(Path(__file__).resolve())
        with self.assertRaises(ValueError):
            self.run_fusion()

    def test_old_geometry_cannot_attach_to_replaced_video(self):
        self.geometry["frames"][0]["video_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different source"):
            self.run_fusion()

    def test_missing_later_geometry_still_ages_the_map(self):
        self.manifest["frames"].append({
            **self.manifest["frames"][0], "frame_id": "f1", "capture_time_s": 7.5,
        })
        result = self.run_fusion()
        self.assertGreater(result["map"]["status_counts"]["expired"], 0)
        self.assertEqual(result["map"]["status_counts"]["consistent"], 0)


if __name__ == "__main__":
    unittest.main()
