"""Generate region-proposal candidates with a local offline vision-language model.

The model only produces a single-turn JSON answer with rectangles in 0-1000 relative
coordinates (Qwen3-VL grounding convention), which are rescaled to absolute image
pixels and strictly validated. Nothing in the model text is executed, merged into the
navigation stack, or treated as a safety decision. Raw responses are archived for audit
next to the proposal document so assistant/human candidates are never confused with
VLM output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np

from field_boundary_mapping.evidence.review_results import ResultStore
from field_boundary_mapping.proposals.run_offline import write_json
from field_boundary_mapping.proposals.run_proposals import REGION_PROMPTS, fields, identifier

BASE = Path(__file__).resolve().parents[2]   # 仓库根
PROMPT_VERSION = "vlm-proposals-v4"
MAX_PROPOSALS_PER_FRAME = 8
MAX_NEW_TOKENS = 1024

INSTRUCTION = """You are looking at one image from a camera mounted on an agricultural vehicle.
Propose up to 6 rectangular region candidates for semantic segmentation review.

Allowed "semantic" values (no others): crop, harvested_ground, ground, water, ditch, vehicle, person, tarp, woody_vegetation, pole.
"box_xyxy" is [x0, y0, x1, y1]: integers in 0-1000 RELATIVE coordinates, where 0 is the left/top edge and 1000 is the right/bottom edge of the image, with 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000.
"relation" MUST be "unspecified" for every proposal, EXCEPT exactly one proposal whose semantic is "vehicle" and whose box covers the visible body of the vehicle the camera is mounted on (hood, tires); that single vehicle proposal uses "ego_candidate". Never write "ego_candidate" for crop, ground, water, ditch, trees, or any non-vehicle region. Proposals for scene regions like crops, water, ditches, or trees are NOT the mounted vehicle.
Do not repeat the same semantic and box twice.
"rationale" is one short sentence describing where the region is. Never claim certainty, verified semantics, safe turning areas, or that any obstacle should be removed or ignored.

Answer with JSON only, no extra text:
{"proposals": [{"semantic": "...", "box_xyxy": [x0, y0, x1, y1], "relation": "...", "rationale": "..."}]}
If no region of the allowed types is visible, answer {"proposals": []}."""


def _absolute(value, size):
    """Rescale one 0-1000 relative coordinate (Qwen3-VL grounding convention) to a pixel bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return None
    if not 0.0 <= float(value) <= 1000.0:
        return None
    return int(round(float(value) * size / 1000.0))


def parse_proposals(text, width, height):
    """Return (accepted, rejected): validated boxes and audit records, never silently repaired.

    A proposal whose scaled pixel box exactly equals an already accepted proposal is
    rejected as a duplicate (reason "duplicate_box_of:<id>"). Near-overlapping boxes stay:
    they may be genuine semantic ambiguity and belong to review, not to the parser.
    """
    accepted, rejected = [], []
    seen_boxes = {}
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return [], [{"reason": "no_json_object", "raw": text}]
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], [{"reason": f"json_decode_error: {exc.msg}", "raw": text}]
    if not isinstance(data, dict) or not isinstance(data.get("proposals"), list):
        return [], [{"reason": "missing_proposals_array", "raw": text}]
    for index, item in enumerate(data["proposals"]):
        if len(accepted) >= MAX_PROPOSALS_PER_FRAME:
            rejected.append({"index": index, "reason": "budget_exceeded", "raw": item})
            continue
        if not isinstance(item, dict) or set(item) != {"semantic", "box_xyxy", "relation", "rationale"}:
            rejected.append({"index": index, "reason": "unexpected_fields", "raw": item})
            continue
        semantic = item["semantic"]
        if semantic not in REGION_PROMPTS:
            rejected.append({"index": index, "reason": f"unsupported_semantic:{semantic}", "raw": item})
            continue
        relation = item["relation"]
        if relation not in ("unspecified", "ego_candidate") or (relation == "ego_candidate" and semantic != "vehicle"):
            rejected.append({"index": index, "reason": "invalid_relation", "raw": item})
            continue
        box = item["box_xyxy"]
        if not isinstance(box, list) or len(box) != 4:
            rejected.append({"index": index, "reason": "box_shape", "raw": item})
            continue
        x0, y0, x1, y1 = (_absolute(value, size) for value, size in zip(box, (width, height, width, height)))
        if x0 is None or y0 is None or x1 is None or y1 is None:
            rejected.append({"index": index, "reason": "box_coordinate_invalid", "raw": item})
            continue
        if not (x0 < x1 and y0 < y1):
            rejected.append({"index": index, "reason": "box_empty_after_scaling", "raw": item})
            continue
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
            rejected.append({"index": index, "reason": "rationale_length", "raw": item})
            continue
        scaled = (x0, y0, x1, y1)
        if scaled in seen_boxes:
            rejected.append({"index": index, "reason": f"duplicate_box_of:{seen_boxes[scaled]}", "raw": item})
            continue
        proposal_id = f"vlm-{len(accepted):02d}"
        seen_boxes[scaled] = proposal_id
        accepted.append({
            "proposal_id": proposal_id, "semantic": semantic,
            "box_xyxy": [x0, y0, x1, y1], "relation": relation,
            "rationale": rationale.strip(),
            "model_box_raw": box,
        })
    return accepted, rejected


class LocalVLM:
    def __init__(self, model_dir, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_dir = Path(model_dir).resolve(strict=True)
        required = ("config.json", "preprocessor_config.json")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Local model files missing: {missing}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch reports no available GPU")
        tensor_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_dir, torch_dtype=tensor_dtype, local_files_only=True,
        ).to(device)
        self.model.eval()
        self.metadata = {
            "backend": "local_qwen3_vl", "model_dir": str(self.model_dir),
            "model_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(self.model_dir.glob("*.json"))},
            "dtype": dtype, "device": device,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "prompt_version": PROMPT_VERSION, "vlm_used": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "score_meaning": "none; VLM emits text boxes only, no calibration and no safety probability",
        }
        print(f"model loaded on {device} in {time.perf_counter() - started:.1f}s", flush=True)

    def ask(self, image_bytes):
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": INSTRUCTION},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = output[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(trimmed, skip_special_tokens=True)
        return response, {"seconds": time.perf_counter() - started,
                          "prompt_tokens": int(inputs["input_ids"].shape[1]),
                          "generated_tokens": int(trimmed.shape[0])}


def run(args):
    store = ResultStore(args.results)
    manifest_path = store.resolve("manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    frames = {}
    for frame in manifest["frames"]:
        identifier(frame.get("frame_id"), "source frame_id")
        if frame["frame_id"] in frames:
            raise ValueError("Duplicate source frame_id")
        frames[frame["frame_id"]] = frame
    selected = []
    for frame_id in args.frames:
        identifier(frame_id, "frame_id")
        if frame_id not in frames:
            raise ValueError(f"Unknown source frame_id: {frame_id}")
        selected.append(frames[frame_id])
    if not 1 <= len(selected) <= 16:
        raise ValueError("Select 1 to 16 frames explicitly")

    model = LocalVLM(args.model_dir, args.device, args.dtype)
    document = {"schema_version": "field_boundary_proposals_v1", "proposal_source": "local_vlm", "frames": []}
    audit = {"prompt_version": PROMPT_VERSION, "model": model.metadata, "instruction": INSTRUCTION, "frames": []}
    for frame in selected:
        frame_id = frame["frame_id"]
        image_bytes, _ = store.read_file(frame["image_path"])
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != frame.get("artifact_sha256", {}).get(frame["image_path"]):
            raise ValueError(f"Source manifest hash mismatch for {frame_id}")
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Cannot decode image for {frame_id}")
        height, width = decoded.shape[:2]
        print(f"vlm inference {frame_id} ({width}x{height})", flush=True)
        response, timing = model.ask(image_bytes)
        proposals, rejected = parse_proposals(response, width, height)
        print(f"  {len(proposals)} accepted, {len(rejected)} rejected", flush=True)
        document["frames"].append({
            "frame_id": frame_id, "image_sha256": digest,
            "proposals": [{key: value for key, value in item.items() if key != "model_box_raw"} for item in proposals],
        })
        audit["frames"].append({
            "frame_id": frame_id, "image_sha256": digest, "image_width": width, "image_height": height,
            "raw_response": response, "timing": timing,
            "accepted": proposals, "rejected": rejected,
        })
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Proposal output must be a new file; existing files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, document)
    write_json(output.with_name(output.stem + "_raw_vlm_outputs.json"), audit)
    print(f"saved {output} and raw outputs; {sum(len(f['proposals']) for f in document['frames'])} proposals", flush=True)
    return document


def main():
    parser = argparse.ArgumentParser(description="Local offline VLM region proposals; JSON boxes only, no tool execution")
    parser.add_argument("--results", type=Path, required=True, help="Existing source run with hashed original images")
    parser.add_argument("--frames", nargs="+", required=True, help="1 to 16 explicit source frame_ids")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local Qwen3-VL model directory")
    parser.add_argument("--output", type=Path, required=True, help="New proposals JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
