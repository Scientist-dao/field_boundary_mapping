from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_offline import evaluate
from run_offline import write_json


class EvaluationTests(unittest.TestCase):
    def test_unknown_is_abstention_not_perfect_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", {"frames": [{
                "frame_id": "f0", "video_name": "test.mp4", "sample_second": 0,
                "evidence_path": "evidence.json",
            }]})
            write_json(root / "evidence.json", {
                "stage_prediction": "UNKNOWN", "hypothesis": {"hypothesis": "headland_or_field_road"},
                "hazards": [{"semantic": "person"}],
            })
            reference = {"reviewed": False, "videos": {"test.mp4": {
                "stages": ["HEADLAND"], "uncertain_seconds": [0],
                "hazards": [{"type": "person", "t_start": 0, "t_end": 1}],
            }}}
            write_json(root / "labels.json", reference)
            before = (root / "labels.json").read_bytes()
            report = evaluate(root, root / "labels.json")
            self.assertEqual(report["stage_coverage"], 0)
            self.assertIsNone(report["stage_accuracy_on_covered_frames"])
            self.assertEqual(report["coarse_hazard_presence_counts"]["person"]["tp"], 1)
            self.assertEqual(report["uncertain_reference_frames"], 1)
            self.assertEqual(before, (root / "labels.json").read_bytes())

    def test_empty_run_has_no_fabricated_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", {"frames": []})
            write_json(root / "labels.json", {"reviewed": False, "videos": {}})
            report = evaluate(root, root / "labels.json")
            self.assertIsNone(report["stage_coverage"])
            self.assertIsNone(report["stage_accuracy_on_covered_frames"])


if __name__ == "__main__":
    unittest.main()
