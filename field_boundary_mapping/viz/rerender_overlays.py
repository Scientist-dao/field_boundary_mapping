"""Regenerate overlay images from stored masks and evidence; never runs SAM3 inference.

Baseline-style frames (no proposal_views) get their overlay.jpg redrawn as before.
Proposal-run frames additionally get *_passed.jpg clean-view variants (spatially
consistent instances plus box_mask_disagreement flagged masks in orange; only
low-confidence rejected noise is folded). Existing overlay.jpg / proposal_*.jpg,
evidence and masks are never rewritten for proposal runs; rejected masks stay fully
retained in the data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from field_boundary_mapping.proposals.run_proposals import render_candidates
from field_boundary_mapping.semantic_pipeline import render_overlay


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_image_file(target, image):
    # cv2.imwrite cannot open non-ASCII paths on Windows; encode manually.
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {target}")
    target.write_bytes(encoded.tobytes())


def read_image(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def add_proposal_passed_variants(root, item):
    folder = root / item["frame_id"]
    image = read_image(root / item["image_path"])
    evidence = json.loads((folder / "evidence.json").read_text(encoding="utf-8-sig"))
    queries = evidence.get("proposals", [])
    instances = []
    with np.load(folder / "masks.npz") as data:
        for record in evidence.get("instances", []):
            key = record.get("mask_key")
            if key not in data.files:
                raise RuntimeError(f"Mask {key} missing from masks.npz for {item['frame_id']}")
            instances.append({**record, "mask": data[key].astype(bool)})
    passed_path = f"{item['frame_id']}/overlay_passed.jpg"
    write_image_file(folder / "overlay_passed.jpg", render_candidates(image, instances, queries, clean=True))
    item["overlay_passed_path"] = passed_path
    item["artifact_sha256"][passed_path] = file_hash(folder / "overlay_passed.jpg")
    for view in item.get("proposal_views", []):
        proposal_id = view["proposal_id"]
        query = next((entry for entry in queries if entry["proposal_id"] == proposal_id), None)
        if query is None:
            raise RuntimeError(f"No proposal record for view {proposal_id} in {item['frame_id']}")
        selected = [instance for instance in instances if instance["proposal_id"] == proposal_id]
        relative = f"{item['frame_id']}/proposal_{proposal_id}_passed.jpg"
        write_image_file(folder / f"proposal_{proposal_id}_passed.jpg",
                         render_candidates(image, selected, [query], clean=True))
        view["overlay_passed_path"] = relative
        item["artifact_sha256"][relative] = file_hash(root / relative)
    print(f"passed variants for {item['frame_id']}", flush=True)


def rerender_baseline_overlay(root, item):
    folder = root / item["frame_id"]
    image = read_image(folder / "frame.jpg")
    with np.load(folder / "masks.npz") as data:
        masks = {key: data[key].astype(bool) for key in data.files
                 if key not in ("surface_labels", "surface_conflict")}
        conflict = data["surface_conflict"].astype(bool)
    evidence = json.loads((folder / "evidence.json").read_text(encoding="utf-8-sig"))
    lines = [poly for boundary in evidence.get("boundaries", []) for poly in boundary["polylines_px"]]
    target = folder / "overlay.jpg"
    write_image_file(target, render_overlay(image, masks, conflict, lines))
    item["artifact_sha256"][item["overlay_path"]] = file_hash(target)
    print(f"rerendered {item['frame_id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="用已保存的掩码重画叠加图/补生成干净视图变体，不重新推理")
    parser.add_argument("--results", required=True, type=Path, help="包含 manifest.json 的离线结果目录")
    args = parser.parse_args()
    root = args.results.resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for item in manifest["frames"]:
        if "proposal_views" in item:
            add_proposal_passed_variants(root, item)
        else:
            rerender_baseline_overlay(root, item)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
