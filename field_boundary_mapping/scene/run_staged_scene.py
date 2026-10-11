import argparse
import hashlib
import json
from pathlib import Path

from field_boundary_mapping.perception.cloud_vlm_proposer import CloudVLM, DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from field_boundary_mapping.evidence.review_results import ResultStore
from field_boundary_mapping.proposals.run_offline import file_hash, write_json
from field_boundary_mapping.proposals.run_proposals import identifier
from field_boundary_mapping.scene.run_scene_understanding import prepare_contexts, read_frame, source_frames
from field_boundary_mapping.scene.scene_staging import (ASSIGNMENT_INSTRUCTION, CATALOGUE_INSTRUCTION, INVENTORY_INSTRUCTION,
                           INVENTORY_VERSION, LOCALIZATION_INSTRUCTION, STAGED_VERSION,
                           assemble_scene, assignment_instruction, bind_inventory, localization_audit,
                           localization_instruction, parse_assignments, parse_catalogue,
                           parse_inventory, parse_localization)
from field_boundary_mapping.scene.scene_understanding import GATES


BASE = Path(__file__).resolve().parents[2]   # 仓库根
PKG = BASE / "field_boundary_mapping"        # Python 包根
RUN_SCHEMA = "field_functional_scene_staged_run_v1"
RECORD_SCHEMA = "field_functional_scene_staged_record_v1"
RESPONSE_FORMAT = {"type": "json_object"}


def catalogue_instruction_for(context, inventory=None):
    target_time = context[-1]["capture_time_s"]
    order = "; ".join(
        f"image {i + 1}: {'CURRENT' if i == len(context) - 1 else 'PAST'}, "
        f"relative time {frame['capture_time_s'] - target_time:.3f}s"
        for i, frame in enumerate(context))
    instruction = CATALOGUE_INSTRUCTION if inventory is None else assignment_instruction(inventory)
    return instruction + "\nImage order: " + order + ". Catalogue CURRENT only."


def template_hashes(pipeline_version=STAGED_VERSION):
    if pipeline_version == STAGED_VERSION:
        templates = {"catalogue": CATALOGUE_INSTRUCTION, "localization": LOCALIZATION_INSTRUCTION}
    elif pipeline_version == INVENTORY_VERSION:
        templates = {"inventory": INVENTORY_INSTRUCTION, "catalogue": ASSIGNMENT_INSTRUCTION,
                     "localization": LOCALIZATION_INSTRUCTION}
    else:
        raise ValueError("Unsupported staged pipeline version")
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in templates.items()}


def stage_identity(stage, name, context, instruction):
    if (stage.get("stage") != name or stage.get("context") != context
            or stage.get("instruction") != instruction or stage.get("response_format") != RESPONSE_FORMAT
            or stage.get("strict_content") is not True
            or any(stage.get(key) is not False for key in GATES)):
        raise ValueError("Staged request identity mismatch")


def completed_stage(stage):
    timing = stage.get("timing", {})
    if (stage.get("status") != "parsed" or timing.get("finish_reason") != "stop"
            or timing.get("parsed_from") != "content" or timing.get("strict_content") is not True):
        raise ValueError("Staged response must be complete original final content")
    return stage["raw_response"]


def check_staged_record(record, frame_id, context, pipeline_version=STAGED_VERSION):
    expected_stages = set(template_hashes(pipeline_version))
    if (record.get("schema_version") != RECORD_SCHEMA or record.get("frame_id") != frame_id
            or record.get("context") != context or record.get("status") != "parsed"
            or record.get("pipeline_version") != pipeline_version
            or any(record.get(key) is not False for key in GATES)):
        raise ValueError("Staged record identity or status mismatch")
    stages = record["stages"]
    if set(stages) != expected_stages:
        raise ValueError("All pipeline stages are required")
    inventory = None
    if pipeline_version == INVENTORY_VERSION:
        initial = stages["inventory"]
        stage_identity(initial, "inventory", context[-1:], INVENTORY_INSTRUCTION)
        inventory = parse_inventory(completed_stage(initial))
        if initial.get("parsed") != inventory:
            raise ValueError("Inventory does not match raw response")
    first, second = stages["catalogue"], stages["localization"]
    stage_identity(first, "catalogue", context, catalogue_instruction_for(context, inventory))
    if inventory is None:
        parsed = catalogue = parse_catalogue(completed_stage(first))
    else:
        if not inventory["items"]:
            if first.get("status") != "skipped_no_inventory" or "raw_response" in first or "timing" in first:
                raise ValueError("Empty inventory requires an explicit skipped assignment")
            parsed = {"assignments": [], "relations": [], "boundaries": [], "unknowns": []}
        else:
            parsed = parse_assignments(completed_stage(first), inventory)
        catalogue = bind_inventory(inventory, parsed)
        if record.get("catalogue") != catalogue:
            raise ValueError("Derived catalogue does not match inventory and assignments")
    if first.get("parsed") != parsed:
        raise ValueError("Catalogue stage does not match raw response")
    stage_identity(second, "localization", context[-1:], localization_instruction(catalogue))
    if not catalogue["regions"]:
        if second.get("status") != "skipped_no_targets" or "raw_response" in second or "timing" in second:
            raise ValueError("Empty catalogue requires an explicit skipped localization")
        localized = {"regions": [], "boundaries": []}
    else:
        localized = parse_localization(completed_stage(second), catalogue)
    if second.get("parsed") != localized:
        raise ValueError("Localization does not match raw response")
    scene = assemble_scene(catalogue, localized)
    if record.get("scene") != scene or record.get("localization_audit") != localization_audit(catalogue, localized):
        raise ValueError("Derived scene or localization audit mismatch")
    return scene


def run(args):
    if args.base_url.rstrip("/") != DEFAULT_BASE_URL or args.api_key_env != DEFAULT_API_KEY_ENV:
        raise ValueError("Use the authorized DashScope endpoint and DASHSCOPE_API_KEY")
    if not 1 <= args.max_output_tokens <= 8192 or not 0 < args.timeout <= 3600:
        raise ValueError("Output tokens must be 1..8192 and timeout 0..3600 seconds")
    templates = template_hashes(args.pipeline_version)
    store = ResultStore(args.results)
    frames = source_frames(store)
    if not 1 <= len(args.frames) <= 16 or len(set(args.frames)) != len(args.frames):
        raise ValueError("Select 1 to 16 distinct target frames")
    for frame_id in args.frames:
        identifier(frame_id, "target frame ID")
        if frame_id not in frames:
            raise ValueError("Unknown target frame")
    prepared = prepare_contexts(store, frames, args.frames, args.mode, args.history_frames, args.history_seconds)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(store.root):
        raise ValueError("Staged output must be new and outside the read-only source run")
    configuration = {
        "pipeline_version": args.pipeline_version, "model": args.model, "base_url": DEFAULT_BASE_URL,
        "mode": args.mode, "history_frames": args.history_frames, "history_seconds": args.history_seconds,
        "max_output_tokens": args.max_output_tokens, "response_format": RESPONSE_FORMAT, "strict_content": True,
        "target_frames": args.frames, "source_manifest_sha256": file_hash(store.resolve("manifest.json")),
        "template_sha256": templates,
        "implementation_sha256": {path.name: file_hash(path) for path in (
            Path(__file__), PKG / "scene" / "scene_staging.py",
            PKG / "scene" / "scene_understanding.py",
            PKG / "scene" / "run_scene_understanding.py",
            PKG / "perception" / "cloud_vlm_proposer.py")},
    }
    model = CloudVLM(DEFAULT_BASE_URL, args.model, DEFAULT_API_KEY_ENV, args.max_output_tokens, args.timeout)
    manifest = {"schema_version": RUN_SCHEMA, "configuration": configuration, "status": "running",
                "purpose": "development_staged_hypotheses_not_held_out_accuracy",
                "cloud_reproducibility": "archived_responses_not_bit_reproducible", **GATES, "frames": []}
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    for frame_id, context, images in prepared:
        path = output / (frame_id + ".json")
        record = {"schema_version": RECORD_SCHEMA, "pipeline_version": args.pipeline_version,
                  "frame_id": frame_id, "context": context, "status": "requesting", "stages": {}, **GATES}
        inventory, catalogue, stage = None, None, None
        try:
            for name in templates:
                if name == "inventory":
                    instruction, response_key = INVENTORY_INSTRUCTION, "items"
                elif name == "catalogue":
                    instruction = catalogue_instruction_for(context, inventory)
                    response_key = "regions" if inventory is None else "assignments"
                else:
                    instruction, response_key = localization_instruction(catalogue), "regions"
                stage_context = context if name == "catalogue" else context[-1:]
                stage_images = images if name == "catalogue" else images[-1:]
                stage = {"stage": name, "context": stage_context, "instruction": instruction,
                         "response_format": RESPONSE_FORMAT, "strict_content": True, "status": "requesting", **GATES}
                record["stages"][name] = stage
                write_json(path, record)
                if name == "catalogue" and inventory is not None and not inventory["items"]:
                    parsed = {"assignments": [], "relations": [], "boundaries": [], "unknowns": []}
                    stage.update(status="skipped_no_inventory", parsed=parsed)
                    catalogue = bind_inventory(inventory, parsed)
                    record["catalogue"] = catalogue
                    write_json(path, record)
                    continue
                if name == "localization" and not catalogue["regions"]:
                    stage.update(status="skipped_no_targets", parsed={"regions": [], "boundaries": []})
                    write_json(path, record)
                    continue
                print(f"staged {args.mode} {frame_id}: {name}; {len(stage_images)} image(s)", flush=True)
                raw, timing = model.ask_images(stage_images, instruction, response_key=response_key,
                                               response_format=RESPONSE_FORMAT, strict_content=True)
                stage.update(raw_response=raw, timing=timing, status="response_received")
                write_json(path, record)
                if (timing.get("finish_reason") != "stop" or timing.get("parsed_from") != "content"
                        or timing.get("strict_content") is not True):
                    raise ValueError("Provider response must be complete original final content")
                if name == "inventory":
                    parsed = inventory = parse_inventory(raw)
                elif name == "catalogue":
                    if inventory is None:
                        parsed = catalogue = parse_catalogue(raw)
                    else:
                        parsed = parse_assignments(raw, inventory)
                        catalogue = bind_inventory(inventory, parsed)
                        record["catalogue"] = catalogue
                else:
                    parsed = parse_localization(raw, catalogue)
                stage.update(parsed=parsed, status="parsed")
                write_json(path, record)
            localized = record["stages"]["localization"]["parsed"]
            record.update(scene=assemble_scene(catalogue, localized),
                          localization_audit=localization_audit(catalogue, localized), status="parsed")
            check_staged_record(record, frame_id, context, args.pipeline_version)
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
            if stage is not None and stage["status"] in ("requesting", "response_received"):
                stage.update(status="error", error_type=type(exc).__name__)
            write_json(path, record)
            manifest["frames"].append({"frame_id": frame_id, "sha256": file_hash(path)})
            manifest["status"] = "incomplete"
            write_json(manifest_path, manifest)
            raise
        write_json(path, record)
        manifest["frames"].append({"frame_id": frame_id, "sha256": file_hash(path)})
        write_json(manifest_path, manifest)
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    return manifest


def load_staged_scene_run(results, scenes):
    store = ResultStore(results)
    manifest = json.loads((Path(scenes) / "manifest.json").read_bytes())
    if (manifest.get("schema_version") != RUN_SCHEMA or manifest.get("status") != "completed"
            or any(manifest.get(key) is not False for key in GATES)):
        raise ValueError("A completed unverified staged run is required")
    config = manifest["configuration"]
    pipeline_version = config["pipeline_version"]
    if (config.get("template_sha256") != template_hashes(pipeline_version)
            or config.get("base_url") != DEFAULT_BASE_URL or config.get("response_format") != RESPONSE_FORMAT
            or config.get("strict_content") is not True):
        raise ValueError("Staged configuration or prompt mismatch")
    if file_hash(store.resolve("manifest.json")) != config["source_manifest_sha256"]:
        raise ValueError("Staged source manifest mismatch")
    selected = config["target_frames"]
    if not 1 <= len(selected) <= 16 or len(set(selected)) != len(selected):
        raise ValueError("Invalid staged target frames")
    if [entry["frame_id"] for entry in manifest["frames"]] != selected:
        raise ValueError("Staged manifest missing or reordered target records")
    frames = source_frames(store)
    for frame_id in selected:
        identifier(frame_id, "staged target frame ID")
        if frame_id not in frames:
            raise ValueError("Unknown staged target frame")
    prepared = prepare_contexts(store, frames, selected, config["mode"], config["history_frames"], config["history_seconds"])
    output = []
    for entry, (frame_id, context, _) in zip(manifest["frames"], prepared):
        path = Path(scenes) / (frame_id + ".json")
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Staged record hash mismatch")
        record = json.loads(path.read_bytes())
        scene = check_staged_record(record, frame_id, context, pipeline_version)
        metadata, image_bytes, image = read_frame(store, frames[frame_id])
        output.append({"frame": frames[frame_id], "scene": scene, "record": record,
                       "image": image, "image_bytes": image_bytes, "image_sha256": metadata["image_sha256"],
                       "scene_record_sha256": entry["sha256"]})
    return manifest, output


def main():
    parser = argparse.ArgumentParser(description="Experimental semantic catalogue then CURRENT-only localization; no JSON repair")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "causal"), required=True)
    parser.add_argument("--history-frames", type=int, default=2)
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--pipeline-version", choices=(STAGED_VERSION, INVENTORY_VERSION), default=STAGED_VERSION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
