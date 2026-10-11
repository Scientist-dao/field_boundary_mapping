import importlib.metadata
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np


PROMPT_VERSION = "field-observables-v1"
PROMPTS = (
    ("crop", "crop plants"),
    ("ground", "bare soil"),
    ("ground", "dirt road"),
    ("woody_vegetation", "trees and bushes"),
    ("water", "water"),
    ("ditch", "drainage ditch"),
    ("person", "person"),
    ("vehicle", "vehicle"),
    ("tarp", "plastic sheet"),
    ("pole", "utility pole"),
)


@contextmanager
def offline_runtime():
    def denied(*args, **kwargs):
        raise RuntimeError("Network access is disabled for local company-data inference")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false"}))
        stack.enter_context(patch.object(socket, "getaddrinfo", denied))
        for method in ("connect", "connect_ex", "send", "sendall", "sendto", "sendfile", "sendmsg"):
            if hasattr(socket.socket, method):
                stack.enter_context(patch.object(socket.socket, method, denied))
        yield


class LocalSAM3:
    def __init__(self, weights, clip_dir=None, confidence=0.3, device="cpu", threads=4):
        weights = Path(weights).resolve(strict=True)
        if not weights.is_file():
            raise ValueError("weights must be an existing local file")
        clip_dir = Path(clip_dir or weights.parent).resolve(strict=True)
        if not (clip_dir / "clip" / "simple_tokenizer.py").is_file():
            raise FileNotFoundError("Local CLIP tokenizer is missing; no automatic installation is permitted")
        for name in ("ultralytics", "torch", "torchvision", "ftfy", "regex"):
            if importlib.util.find_spec(name) is None:
                raise RuntimeError(f"Missing local dependency: {name}; use an existing SAM3 environment")
        if not 0 < confidence < 1 or threads < 1:
            raise ValueError("confidence must be in (0,1), threads must be positive")
        sys.path.insert(0, str(clip_dir))
        import torch
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        torch.set_num_threads(threads)
        self.predictor = SAM3SemanticPredictor(overrides={
            "model": str(weights), "conf": confidence, "imgsz": 1008,
            "device": device, "task": "segment", "mode": "predict",
            "verbose": False, "save": False,
        })
        self.metadata = {
            "backend": "local_sam3_fixed_prompts", "vlm_used": False,
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "torch_version": torch.__version__, "weights_name": weights.name,
            "weights_size_bytes": weights.stat().st_size,
            "weights_mtime_ns": weights.stat().st_mtime_ns,
            "prompt_version": PROMPT_VERSION,
            "prompts": [{"semantic": s, "text": p} for s, p in PROMPTS],
            "confidence_threshold": confidence, "device": device, "threads": threads,
            "score_meaning": "uncalibrated_model_score_not_safety_probability",
        }

    def predict(self, image):
        return self._predict_queries(image, [
            {"semantic": semantic, "prompt": prompt} for semantic, prompt in PROMPTS
        ])

    def predict_regions(self, image, queries):
        if not 1 <= len(queries) <= 8:
            raise ValueError("Region inference requires 1 to 8 queries per image")
        height, width = image.shape[:2]
        for query in queries:
            box = query["box_xyxy"]
            if (len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError("Region query box must be an in-image integer xyxy rectangle")
            if not isinstance(query["prompt"], str) or not query["prompt"].strip():
                raise ValueError("Region query prompt must be nonempty")
        return self._predict_queries(image, queries)

    def _predict_queries(self, image, queries):
        started = time.perf_counter()
        self.predictor.reset_image()
        self.predictor.set_image(image)
        encoder_seconds = time.perf_counter() - started
        detections = []
        timings = []
        for query in queries:
            semantic, prompt = query["semantic"], query["prompt"]
            geometry = {}
            if "box_xyxy" in query:
                geometry = {"bboxes": [query["box_xyxy"]], "labels": [1]}
            t0 = time.perf_counter()
            masks, boxes = self.predictor.inference_features(
                self.predictor.features, image.shape[:2], text=[prompt], **geometry,
            )
            provenance = {key: query[key] for key in ("proposal_id", "box_xyxy") if key in query}
            if masks is not None:
                masks = masks.detach().cpu().numpy().astype(bool)
                boxes = boxes.detach().cpu().numpy()
                for mask, box in zip(masks, boxes):
                    if mask.shape != image.shape[:2]:
                        raise RuntimeError("SAM3 mask coordinates do not match the original image")
                    detections.append({
                        "semantic": semantic, "prompt": prompt, **provenance,
                        "score": float(box[4]), "bbox_xyxy": box[:4].tolist(),
                        "mask": mask,
                    })
            timings.append({"prompt": prompt, **provenance, "seconds": time.perf_counter() - t0})
            print(f"  {prompt}: {timings[-1]['seconds']:.1f}s", flush=True)
        return detections, {
            "encoder_seconds": encoder_seconds, "prompts": timings,
            "total_seconds": time.perf_counter() - started,
        }
