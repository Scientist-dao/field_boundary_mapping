import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_offline import VideoReader, run, sample_requests, write_json
from sam3_local import offline_runtime
from semantic_pipeline import analyze_frame, interface_polylines, load_core, mask_contours


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def setUp(self):
        self.image = np.zeros((20, 30, 3), dtype=np.uint8)
        self.crop = np.zeros((20, 30), dtype=bool)
        self.crop[:, :15] = True
        self.ground = ~self.crop

    def detection(self, name, mask, score=0.8):
        return {"semantic": name, "mask": mask, "score": score, "prompt": name}

    def analyze(self, detections, **kwargs):
        return analyze_frame(self.image, detections, "test-frame", self.core, **kwargs)

    def test_no_masks_is_unknown_not_free(self):
        evidence, _, _ = self.analyze([])
        self.assertEqual(evidence["unknown_fraction"], 1)
        self.assertEqual(evidence["hypothesis"]["hypothesis"], "unknown")
        self.assertIsNone(evidence["world_map"])
        self.assertEqual(evidence["stage_prediction"], "UNKNOWN")

    def test_contact_is_candidate_not_physical_boundary(self):
        evidence, _, _ = self.analyze([
            self.detection("crop", self.crop), self.detection("ground", self.ground),
        ])
        self.assertEqual(evidence["hypothesis"]["hypothesis"], "headland_or_field_road")
        self.assertFalse(evidence["hypothesis"]["maneuver_verified"])
        self.assertFalse(evidence["boundaries"][0]["physical_field_boundary_verified"])
        lines = evidence["boundaries"][0]["polylines_px"]
        self.assertEqual(len(lines), 1)
        self.assertNotEqual(lines[0][0], lines[0][-1])

    def test_synonymous_prompts_do_not_create_conflict(self):
        evidence, _, _ = self.analyze([
            self.detection("ground", self.ground), self.detection("ground", self.ground, 0.7),
        ])
        self.assertEqual(evidence["conflict_fraction"], 0)
        self.assertEqual(len(evidence["regions"]), 1)
        self.assertEqual(evidence["regions"][0]["score"], 0.8)

    def test_conflicting_surfaces_do_not_vote_ground(self):
        evidence, _, arrays = self.analyze([
            self.detection("crop", self.crop), self.detection("ground", self.crop),
        ])
        self.assertEqual(evidence["conflict_fraction"], 0.5)
        self.assertFalse(arrays["surface_labels"].any())
        self.assertFalse(evidence["boundaries"])

    def test_obstacle_is_separate_and_occludes_ground(self):
        evidence, _, arrays = self.analyze([
            self.detection("ground", self.ground), self.detection("person", self.ground),
        ])
        self.assertEqual(evidence["hazards"][0]["semantic"], "person")
        self.assertFalse(arrays["surface_labels"].any())
        self.assertFalse(evidence["hypothesis"]["maneuver_verified"])

    def test_unknown_gap_is_not_bridged(self):
        ground = self.ground.copy()
        ground[:, 15:17] = False
        evidence, _, _ = self.analyze([
            self.detection("crop", self.crop), self.detection("ground", ground),
        ])
        self.assertFalse(evidence["boundaries"])

    def test_mask_hole_preserved(self):
        mask = np.ones((20, 20), dtype=bool)
        mask[5:15, 5:15] = False
        contours = mask_contours(mask)
        self.assertEqual(sorted(c["hole"] for c in contours), [False, True])

    def test_disconnected_edges_remain_separate(self):
        labels = np.zeros((20, 20), dtype=np.uint8)
        labels[:5, :10] = 1
        labels[:5, 10:] = 2
        labels[15:, :10] = 1
        labels[15:, 10:] = 2
        self.assertEqual(len(interface_polylines(labels, 1, 2)), 2)

    def test_invalid_detections_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze([self.detection("ground", self.ground, float("nan"))])
        with self.assertRaises(ValueError):
            self.analyze([self.detection("ground", np.ones((2, 2)))])

    def test_row_continuation_is_counterevidence(self):
        evidence, _, _ = self.analyze([
            self.detection("crop", self.crop), self.detection("ground", self.ground),
        ], row_context={"confirmed_row_ends": 3, "rows_continue": True})
        self.assertEqual(evidence["hypothesis"]["hypothesis"], "field_road_or_gap")

    def test_json_forbids_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            write_json(path, {"valid": True})
            with self.assertRaises(ValueError):
                write_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"valid": True})

    def test_network_guard_restores_socket(self):
        original = socket.socket.connect
        with offline_runtime(), socket.socket() as connection:
            with self.assertRaises(RuntimeError):
                connection.connect(("127.0.0.1", 9))
        self.assertIs(socket.socket.connect, original)

    def test_video_sampling_matches_annotation_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(30):
                writer.write(np.full((24, 32, 3), index * 5, np.uint8))
            writer.release()
            reader = VideoReader(path)
            try:
                image, index, timestamp = reader.read_second(1)
                self.assertEqual((index, timestamp), (15, 1.5))
                self.assertGreater(image.mean(), 65)
                with self.assertRaises(ValueError):
                    reader.read_second(3)
            finally:
                reader.close()
            args = argparse.Namespace(video_dir=Path(directory), sample=["video.mp4:1,1,2"], all=False)
            self.assertEqual(len(sample_requests(args)), 2)

    def test_network_guard_also_blocks_udp_and_dns(self):
        with offline_runtime(), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            with self.assertRaises(RuntimeError):
                connection.sendto(b"synthetic", ("127.0.0.1", 9))
            with self.assertRaises(RuntimeError):
                socket.getaddrinfo("localhost", 80)

    def test_valid_cache_skips_model_and_corrupt_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "synthetic-weights.pt"
            weights.write_bytes(b"not a model: mocked inference fixture")
            video = root / "synthetic.mp4"
            video.write_bytes(b"not a video: mocked reader fixture")
            args = argparse.Namespace(
                weights=weights, output=root / "out", confidence=0.3, device="cpu", threads=1, clip_dir=None,
            )
            with patch("run_offline.sample_requests", return_value=[(video, 0, "synthetic-f0")]), \
                 patch("run_offline.LocalSAM3") as model, patch("run_offline.VideoReader") as reader:
                model.return_value.metadata = {"backend": "synthetic-test-only"}
                model.return_value.predict.return_value = ([], {"total_seconds": 0})
                reader.return_value.read_second.return_value = (self.image, 5, 0.5)
                run(args)
                model.reset_mock()
                run(args)
                model.assert_not_called()
                (root / "out" / "synthetic-f0" / "evidence.json").write_bytes(b"")
                with self.assertRaisesRegex(ValueError, "modified or damaged"):
                    run(args)
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
