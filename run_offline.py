import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from sam3_local import LocalSAM3, PROMPTS, PROMPT_VERSION, offline_runtime
from semantic_pipeline import analyze_frame, load_core


BASE = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_image(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Unable to encode {path.name}")
    path.write_bytes(data.tobytes())


class VideoReader:
    def __init__(self, path):
        self.temporary = None
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            self.temporary = tempfile.TemporaryDirectory(prefix="field-boundary-")
            copy = Path(self.temporary.name) / "video.mp4"
            shutil.copyfile(path, copy)
            self.capture = cv2.VideoCapture(str(copy))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.capture.isOpened() or not math.isfinite(self.fps) or self.fps <= 0 or self.count < 1:
            self.close()
            raise RuntimeError(f"Cannot decode video: {path.name}")

    def read_second(self, second):
        if second < 0 or second >= int(self.count / self.fps):
            raise ValueError(f"Sample second outside video: {second}")
        index = min(round((second + 0.5) * self.fps), self.count - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode requested frame {index}; no timestamp substitution allowed")
        decoded_index = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if decoded_index != index:
            raise RuntimeError(f"Decoder returned frame {decoded_index} instead of {index}")
        return image, index, index / self.fps

    def close(self):
        self.capture.release()
        if self.temporary:
            self.temporary.cleanup()
            self.temporary = None


def sample_requests(args):
    root = args.video_dir.resolve(strict=True)
    selected = {}
    if args.sample:
        for entry in args.sample:
            name, separator, seconds = entry.rpartition(":")
            if not separator:
                raise ValueError("--sample format is video.mp4:second[,second...]")
            selected.setdefault(name, set()).update(int(s) for s in seconds.split(","))
    elif args.all:
        selected = {p.name: None for p in sorted(root.glob("*.mp4"))}
    else:
        raise ValueError("Choose --sample or --all explicitly; full CPU inference is expensive")
    requests = []
    for name, seconds in sorted(selected.items()):
        path = (root / name).resolve(strict=True)
        if path.parent != root or path.suffix.lower() != ".mp4" or Path(name).name != name:
            raise ValueError("Only video basenames directly inside --video-dir are allowed")
        reader = VideoReader(path)
        try:
            selected_seconds = sorted(seconds) if seconds is not None else range(int(reader.count / reader.fps))
            for second in selected_seconds:
                if second < 0 or second >= int(reader.count / reader.fps):
                    raise ValueError(f"Invalid second {second} for {name}")
                index = min(round((second + 0.5) * reader.fps), reader.count - 1)
                requests.append((path, second, f"{path.stem}__f{index:08d}"))
        finally:
            reader.close()
    if not requests:
        raise ValueError("No video samples selected")
    return requests


def run(args):
    requests = sample_requests(args)
    core = load_core()
    weights = args.weights.resolve(strict=True)
    signature = {
        "schema_version": "field_boundary_run_v1", "pipeline_version": "p1.0",
        "weights_sha256": file_hash(weights), "prompt_version": PROMPT_VERSION,
        "prompts": list(PROMPTS), "confidence": args.confidence,
        "device": args.device, "threads": args.threads,
        "implementation_sha256": {
            p.name: file_hash(p) for p in [Path(__file__), BASE / "sam3_local.py", BASE / "semantic_pipeline.py", Path(core.__file__)]
        },
    }
    signature = json.loads(json.dumps(signature))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["configuration"] != signature:
            raise ValueError("Output belongs to different weights/prompts/code; choose a new --output directory")
    else:
        manifest = {
            "schema_version": "field_boundary_run_v1", "configuration": signature,
            "purpose": "local_development_baseline_not_a_held_out_test",
            "annotation_used_for_inference": False, "frames": [],
        }
        write_json(manifest_path, manifest)
    existing = {frame["frame_id"]: frame for frame in manifest["frames"]}
    hashes = {path.name: file_hash(path) for path in {r[0] for r in requests}}
    model = None
    for path, second, frame_id in requests:
        if frame_id in existing:
            item = existing[frame_id]
            if item["video_sha256"] != hashes[path.name]:
                raise ValueError("Source video changed; use a new output directory")
            if all((output / item[key]).is_file() for key in ("image_path", "overlay_path", "evidence_path", "masks_path")):
                for relative_path, digest in item["artifact_sha256"].items():
                    if file_hash(output / relative_path) != digest:
                        raise ValueError("Cached artifact was modified or damaged; choose a new output directory")
                print(f"cached {frame_id}", flush=True)
                continue
        if model is None:
            model = LocalSAM3(weights, args.clip_dir, args.confidence, args.device, args.threads)
        reader = VideoReader(path)
        try:
            image, frame_index, capture_time = reader.read_second(second)
        finally:
            reader.close()
        print(f"infer {frame_id} at {capture_time:.3f}s", flush=True)
        started = time.perf_counter()
        detections, timings = model.predict(image)
        evidence, overlay, arrays = analyze_frame(image, detections, frame_id, core)
        evidence.update({
            "source": {
                "video_name": path.name, "video_sha256": hashes[path.name],
                "frame_index": frame_index, "capture_time_s": capture_time,
                "sample_second": second, "clock": "video_relative_not_rtk_synchronized",
            },
            "model": {**model.metadata, "weights_sha256": signature["weights_sha256"]},
            "timing": timings, "annotation_used_for_inference": False,
            "instances": [{key: value for key, value in d.items() if key != "mask"} for d in detections],
        })
        folder = output / frame_id
        folder.mkdir(exist_ok=True)
        write_image(folder / "frame.jpg", image)
        write_image(folder / "overlay.jpg", overlay)
        np.savez_compressed(folder / "masks.npz", **arrays)
        write_json(folder / "evidence.json", evidence)
        item = {
            "frame_id": frame_id, "video_name": path.name, "video_sha256": hashes[path.name],
            "frame_index": frame_index, "capture_time_s": capture_time, "sample_second": second,
            "image_path": f"{frame_id}/frame.jpg", "overlay_path": f"{frame_id}/overlay.jpg",
            "evidence_path": f"{frame_id}/evidence.json", "masks_path": f"{frame_id}/masks.npz",
        }
        item["artifact_sha256"] = {
            item[key]: file_hash(output / item[key])
            for key in ("image_path", "overlay_path", "evidence_path", "masks_path")
        }
        existing[frame_id] = item
        manifest["frames"] = sorted(existing.values(), key=lambda f: (f["video_name"], f["frame_index"]))
        write_json(manifest_path, manifest)
        print(f"saved {frame_id}; {time.perf_counter() - started:.1f}s; {evidence['hypothesis']['hypothesis']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Local SAM3 + C++ image-evidence baseline; no annotations or network")
    parser.add_argument("--video-dir", type=Path, default=BASE.parent.parent / "地头数据集")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    samples = parser.add_mutually_exclusive_group(required=True)
    samples.add_argument("--sample", action="append", help="video.mp4:0,1,2; repeat for more videos")
    samples.add_argument("--all", action="store_true", help="Process every whole-second bin of every video")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with offline_runtime():
        run(args)


if __name__ == "__main__":
    main()
