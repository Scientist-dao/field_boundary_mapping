import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from field_boundary_mapping.perception.cloud_vlm_proposer import CloudVLM, DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from field_boundary_mapping.evidence.review_results import ResultStore
from field_boundary_mapping.proposals.run_offline import file_hash, write_json
from field_boundary_mapping.proposals.run_proposals import identifier
from field_boundary_mapping.scene.scene_understanding import GATES, INSTRUCTIONS, PROMPT_VERSION, SCHEMA_VERSION, parse_scene


BASE = Path(__file__).resolve().parents[2]   # 仓库根
PKG = BASE / "field_boundary_mapping"        # Python 包根


def source_frames(store):
    manifest = json.loads(store.resolve("manifest.json").read_bytes())
    by_id, seen = {}, set()
    for frame in manifest["frames"]:
        identifier(frame.get("frame_id"), "source frame ID")
        digest = frame.get("video_sha256")
        index, timestamp = frame.get("frame_index"), frame.get("capture_time_s")
        if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                or type(index) is not int or index < 0
                or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Source requires video hash, nonnegative index and timestamp")
        if frame["frame_id"] in by_id or (digest, index) in seen:
            raise ValueError("Duplicate source frame identity")
        by_id[frame["frame_id"]] = frame
        seen.add((digest, index))
    videos = {frame["video_sha256"] for frame in by_id.values()}
    for video in videos:
        ordered = sorted((f for f in by_id.values() if f["video_sha256"] == video), key=lambda f: f["frame_index"])
        if any(a["capture_time_s"] >= b["capture_time_s"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Video frame index and capture time must strictly agree")
    return by_id


def select_context(frames, target_id, mode, history_frames, history_seconds):
    if mode not in ("single", "causal") or type(history_frames) is not int or not 1 <= history_frames <= 3:
        raise ValueError("Use single/causal mode and 1 to 3 history frames")
    if not math.isfinite(history_seconds) or history_seconds <= 0:
        raise ValueError("History duration must be finite and positive")
    target = frames[target_id]
    if mode == "single":
        return [target]
    candidates = [f for f in frames.values() if f["video_sha256"] == target["video_sha256"]
                  and f["frame_index"] < target["frame_index"]
                  and 0 < target["capture_time_s"] - f["capture_time_s"] <= history_seconds]
    candidates.sort(key=lambda f: f["frame_index"])
    return candidates[-history_frames:] + [target]


def read_frame(store, frame):
    data, mime = store.read_file(frame["image_path"])
    digest = hashlib.sha256(data).hexdigest()
    if digest != frame.get("artifact_sha256", {}).get(frame["image_path"]):
        raise ValueError("Source image hash mismatch")
    if mime not in ("image/jpeg", "image/png"):
        raise ValueError("Scene API supports JPEG/PNG source images")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > 16777216:
        raise ValueError("Source must be a decodable image <= 16777216 pixels")
    metadata = {key: frame[key] for key in ("frame_id", "video_sha256", "frame_index", "capture_time_s")}
    metadata.update(image_sha256=digest, width=image.shape[1], height=image.shape[0], mime_type=mime)
    return metadata, data, image


def instruction_for(context, prompt_version=PROMPT_VERSION):
    if prompt_version not in INSTRUCTIONS:
        raise ValueError("Unsupported scene prompt version")
    target_time = context[-1]["capture_time_s"]
    order = "; ".join(f"image {i + 1}: {'CURRENT' if i == len(context) - 1 else 'PAST'}, "
                      f"relative time {frame['capture_time_s'] - target_time:.3f}s"
                      for i, frame in enumerate(context))
    return INSTRUCTIONS[prompt_version] + "\nImage order: " + order + ". Output coordinates for CURRENT only."


def prepare_contexts(store, frames, selected, mode, history_frames, history_seconds):
    prepared = []
    for frame_id in selected:
        context = select_context(frames, frame_id, mode, history_frames, history_seconds)
        loaded = [read_frame(store, frame) for frame in context]
        prepared.append((frame_id, [item[0] for item in loaded], [(item[1], item[0]["mime_type"]) for item in loaded]))
    return prepared


def check_identity(record, frame_id, context, instruction, response_format=None):
    if (record.get("schema_version") != SCHEMA_VERSION or record.get("frame_id") != frame_id
            or record.get("context") != context or record.get("instruction") != instruction
            or record.get("response_format") != response_format
            or any(record.get(key) is not False for key in GATES)):
        raise ValueError("Cached scene identity, context, response format or gates mismatch")


def check_record(record, frame_id, context, instruction, response_format=None):
    check_identity(record, frame_id, context, instruction, response_format)
    if record.get("status") != "parsed":
        raise ValueError("Scene record has no validated complete response")
    if record.get("timing", {}).get("finish_reason") != "stop":
        raise ValueError("Incomplete provider response")
    parsed = parse_scene(record["raw_response"])
    if record.get("scene") != parsed:
        raise ValueError("Parsed scene does not match archived raw response")
    return parsed


def run(args):
    if args.base_url.rstrip("/") != DEFAULT_BASE_URL:
        raise ValueError("Only the authorized DashScope endpoint is supported by this scene runner")
    if args.continue_on_failure and args.resume:
        raise ValueError("--continue-on-failure never resumes: every selected frame gets one request only")
    if not 1 <= args.max_output_tokens <= 8192 or not 0 < args.timeout <= 3600:
        raise ValueError("Output tokens must be 1..8192 and timeout 0..3600 seconds")
    if args.prompt_version not in INSTRUCTIONS:
        raise ValueError("Unsupported scene prompt version")
    if args.response_format not in ("text", "json_object"):
        raise ValueError("Unsupported scene response format")
    response_format = None if args.response_format == "text" else {"type": "json_object"}
    store = ResultStore(args.results)
    frames = source_frames(store)
    if not 1 <= len(args.frames) <= 16 or len(set(args.frames)) != len(args.frames):
        raise ValueError("Select 1 to 16 distinct target frames")
    for frame_id in args.frames:
        identifier(frame_id, "target frame ID")
        if frame_id not in frames:
            raise ValueError("Unknown target frame")
    prepared = prepare_contexts(store, frames, args.frames, args.mode, args.history_frames, args.history_seconds)
    configuration = {
        "prompt_version": args.prompt_version, "mode": args.mode, "model": args.model,
        "base_url": DEFAULT_BASE_URL, "max_output_tokens": args.max_output_tokens,
        "response_format": response_format,
        "history_frames": args.history_frames, "history_seconds": args.history_seconds,
        "source_manifest_sha256": file_hash(store.resolve("manifest.json")),
        "target_frames": args.frames, "instruction_sha256": hashlib.sha256(INSTRUCTIONS[args.prompt_version].encode()).hexdigest(),
        "implementation_sha256": {path.name: file_hash(path) for path in
                                  (Path(__file__), PKG / "scene" / "scene_understanding.py",
                                   PKG / "perception" / "cloud_vlm_proposer.py")},
    }
    output = args.output.resolve()
    if output.is_relative_to(store.root):
        raise ValueError("Scene output must be outside the read-only source run")
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume:
            raise ValueError("Output exists; explicit --resume is required")
        manifest = json.loads(manifest_path.read_bytes())
        if (manifest.get("configuration") != configuration
                or manifest.get("schema_version") != "field_functional_scene_run_v1"
                or any(manifest.get(key) is not False for key in GATES)):
            raise ValueError("Resume configuration mismatch")
        cached_ids = [entry["frame_id"] for entry in manifest["frames"]]
        if cached_ids != args.frames[:len(cached_ids)]:
            raise ValueError("Cached frame records must be a target prefix")
        for entry in manifest["frames"]:
            identifier(entry["frame_id"], "cached frame ID")
            if file_hash(output / (entry["frame_id"] + ".json")) != entry["sha256"]:
                raise ValueError("Cached record hash mismatch")
    else:
        manifest = {"schema_version": "field_functional_scene_run_v1", "configuration": configuration,
                    "status": "running", "purpose": "development_scene_hypotheses_not_held_out_accuracy",
                    "cloud_reproducibility": "archived_responses_not_bit_reproducible", **GATES, "frames": []}
    for attempt in manifest.get("failed_attempts", []):
        identifier(attempt["frame_id"], "failed frame ID")
        if type(attempt["attempt"]) is not int or attempt["attempt"] < 1:
            raise ValueError("Invalid failed attempt index")
        path = output / f"{attempt['frame_id']}.attempt-{attempt['attempt']:04d}.json"
        if file_hash(path) != attempt["sha256"]:
            raise ValueError("Failed attempt archive hash mismatch")
    cached, failed = {}, set()
    for frame_id, context, _ in prepared:
        path = output / (frame_id + ".json")
        if path.exists():
            record = json.loads(path.read_bytes())
            check_identity(record, frame_id, context, instruction_for(context, args.prompt_version), response_format)
            if record.get("status") in ("error", "response_received"):
                failed.add(frame_id)
            else:
                check_record(record, frame_id, context, instruction_for(context, args.prompt_version), response_format)
                cached[frame_id] = record
    model = None
    if len(cached) != len(prepared):
        model = CloudVLM(DEFAULT_BASE_URL, args.model, args.api_key_env, args.max_output_tokens, args.timeout)
    if not output.exists():
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)
    entries = []
    for frame_id, context, images in prepared:
        path = output / (frame_id + ".json")
        instruction = instruction_for(context, args.prompt_version)
        if frame_id in cached:
            record = cached[frame_id]
            print(f"cached scene {frame_id}", flush=True)
        else:
            if frame_id in failed:
                attempt = 1
                archive = output / f"{frame_id}.attempt-{attempt:04d}.json"
                while archive.exists():
                    attempt += 1
                    archive = output / f"{frame_id}.attempt-{attempt:04d}.json"
                path.rename(archive)
                manifest.setdefault("failed_attempts", []).append(
                    {"frame_id": frame_id, "attempt": attempt, "sha256": file_hash(archive)})
                write_json(manifest_path, manifest)
            print(f"scene {args.mode} {frame_id}; {len(images)} image(s)", flush=True)
            record = {"schema_version": SCHEMA_VERSION, "frame_id": frame_id, "context": context,
                      "instruction": instruction, "response_format": response_format, "status": "requesting", **GATES}
            try:
                raw, timing = model.ask_images(images, instruction, response_key="regions", response_format=response_format)
                record.update(raw_response=raw, timing=timing, status="response_received")
                write_json(path, record)
                if timing.get("finish_reason") != "stop":
                    raise ValueError("Provider response incomplete or filtered")
                record["scene"] = parse_scene(raw)
                record["status"] = "parsed"
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                write_json(path, record)
                if not args.continue_on_failure:
                    manifest["status"] = "incomplete"
                    write_json(manifest_path, manifest)
                    raise
                manifest.setdefault("failed_frame_ids", []).append(frame_id)
                manifest["status"] = "incomplete"
                write_json(manifest_path, manifest)
                entries.append({"frame_id": frame_id, "sha256": file_hash(path)})
                manifest.update(frames=list(entries), status="running")
                write_json(manifest_path, manifest)
                continue
            write_json(path, record)
        entries.append({"frame_id": frame_id, "sha256": file_hash(path)})
        manifest.update(frames=list(entries), status="running")
        write_json(manifest_path, manifest)
    manifest["status"] = "completed_with_failures" if manifest.get("failed_frame_ids") else "completed"
    write_json(manifest_path, manifest)
    return manifest


def load_scene_run(results, scenes):
    store = ResultStore(results)
    manifest_path = Path(scenes) / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema_version") == "field_functional_scene_staged_run_v1":
        from field_boundary_mapping.scene.run_staged_scene import load_staged_scene_run
        return load_staged_scene_run(results, scenes)
    if (manifest.get("schema_version") != "field_functional_scene_run_v1"
            or manifest.get("status") not in ("completed", "completed_with_failures")
            or any(manifest.get(key) is not False for key in GATES)):
        raise ValueError("A completed unverified scene run is required")
    failed = manifest.get("failed_frame_ids", [])
    if (manifest.get("status") == "completed") != (not failed):
        raise ValueError("Scene run failure accounting is inconsistent")
    config = manifest["configuration"]
    response_format = config.get("response_format")
    if response_format is not None and (not isinstance(response_format, dict) or response_format != {"type": "json_object"}):
        raise ValueError("Unsupported scene response format")
    prompt_version = config["prompt_version"]
    if (prompt_version not in INSTRUCTIONS or config["instruction_sha256"] !=
            hashlib.sha256(INSTRUCTIONS[prompt_version].encode()).hexdigest()):
        raise ValueError("Scene prompt version or instruction hash mismatch")
    if file_hash(store.resolve("manifest.json")) != config["source_manifest_sha256"]:
        raise ValueError("Scene source manifest mismatch")
    frames = source_frames(store)
    selected = config["target_frames"]
    if not 1 <= len(selected) <= 16 or len(set(selected)) != len(selected):
        raise ValueError("Invalid scene target frames")
    expected = [entry["frame_id"] for entry in manifest["frames"]]
    if expected != selected:
        raise ValueError("Scene manifest missing or reordered target records")
    prepared = prepare_contexts(store, frames, selected, config["mode"], config["history_frames"], config["history_seconds"])
    output, archived_failures = [], []
    for entry, (frame_id, context, _) in zip(manifest["frames"], prepared):
        identifier(frame_id, "scene frame ID")
        path = Path(scenes) / (frame_id + ".json")
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Scene record hash mismatch")
        record = json.loads(path.read_bytes())
        if record.get("status") != "parsed":
            check_identity(record, frame_id, context, instruction_for(context, prompt_version), response_format)
            if record.get("status") != "error" or "scene" in record:
                raise ValueError("Scene record has no validated complete response")
            archived_failures.append(frame_id)
            continue
        scene = check_record(record, frame_id, context, instruction_for(context, prompt_version), response_format)
        metadata, image_bytes, image = read_frame(store, frames[frame_id])
        output.append({"frame": frames[frame_id], "scene": scene, "record": record,
                       "image": image, "image_bytes": image_bytes, "image_sha256": metadata["image_sha256"],
                       "scene_record_sha256": entry["sha256"]})
    if archived_failures != failed:
        raise ValueError("Scene failure accounting does not match the archived records")
    if not output:
        raise ValueError("Every scene record failed to parse, so nothing can be refined")
    return manifest, output


def main():
    parser = argparse.ArgumentParser(description="DashScope functional scene hypotheses; single or past-only context")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "causal"), required=True)
    parser.add_argument("--history-frames", type=int, default=2)
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--prompt-version", choices=tuple(INSTRUCTIONS), default=PROMPT_VERSION)
    parser.add_argument("--response-format", choices=("text", "json_object"), default="text")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="archive unparseable responses as failures and still give each remaining frame "
                             "its single request; never combines with --resume")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
