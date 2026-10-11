"""Render side-by-side comparison panels for ablation runs (read-only).

Each panel shows one frame from one run: spatially consistent instances filled
with their semantic color and white contour; rejected instances drawn as thin
dark-red contours only (they visualise fragmentation without hiding the image).
A legend strip under each panel carries the run label and instance counts, so
the image itself stays text-free.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 640
LEGEND_H = 96
GAP = 12
FONT_CANDIDATES = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf")

SEMANTIC_COLORS = {
    "crop": (64, 168, 96),
    "harvested_ground": (160, 150, 60),
    "ground": (140, 110, 70),
    "water": (70, 60, 200),
    "ditch": (70, 60, 200),
    "vehicle": (40, 40, 230),
    "person": (40, 200, 240),
    "tarp": (200, 80, 220),
    "woody_vegetation": (30, 140, 60),
    "pole": (160, 160, 160),
}
DEFAULT_COLOR = (120, 120, 120)
REJECT_CONTOUR = (60, 60, 200)
FILL_ALPHA = 0.45


def load_font(size):
    for name in FONT_CANDIDATES:
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
    return ImageFont.load_default()


def load_frame(run_dir, frame_id):
    frame_dir = Path(run_dir) / frame_id
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer((frame_dir / "frame.jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    evidence = json.loads((frame_dir / "evidence.json").read_text(encoding="utf-8"))
    masks = np.load(frame_dir / "masks.npz")
    instances = []
    for instance in evidence["instances"]:
        mask = masks[instance["mask_key"]].astype(bool)
        instances.append((instance, mask))
    return image, instances


def render_panel(image, instances):
    overlay = image.astype(np.float64)
    for instance, mask in instances:
        if instance["spatial_check"]["status"] == "spatially_consistent":
            color = np.asarray(SEMANTIC_COLORS.get(instance["semantic"], DEFAULT_COLOR), dtype=np.float64)
            overlay[mask] = (1 - FILL_ALPHA) * overlay[mask] + FILL_ALPHA * color
    overlay = overlay.astype(np.uint8)
    for instance, mask in instances:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if instance["spatial_check"]["status"] == "spatially_consistent":
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        else:
            cv2.drawContours(overlay, contours, -1, REJECT_CONTOUR, 1)
    return overlay


def resize_width(image, width):
    scale = width / image.shape[1]
    return cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def render_comparison(frame_id, runs, labels, output):
    panels = []
    for run in runs:
        image, instances = load_frame(run, frame_id)
        panel = render_panel(image, instances)
        passed = sum(1 for i, _ in instances if i["spatial_check"]["status"] == "spatially_consistent")
        flagged = sum(1 for i, _ in instances if i.get("box_mask_disagreement") is True)
        panels.append((resize_width(panel, PANEL_W), len(instances), passed, flagged))
    height = panels[0][0].shape[0]
    canvas = Image.new("RGB", (PANEL_W * len(panels) + GAP * (len(panels) - 1), height + LEGEND_H), (24, 24, 24))
    font = load_font(24)
    small = load_font(19)
    draw = ImageDraw.Draw(canvas)
    for index, (panel, total, passed, flagged) in enumerate(panels):
        x = index * (PANEL_W + GAP)
        canvas.paste(Image.fromarray(panel), (x, 0))
        cx = x + 16
        draw.text((cx, height + 8), labels[index], fill=(235, 235, 235), font=font)
        draw.text((cx, height + 44), f"instances {total} | passed {passed} | flagged {flagged}", fill=(170, 170, 170), font=small)
    output = Path(output)
    canvas.save(output, quality=92)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render side-by-side ablation comparison panels (read-only)")
    parser.add_argument("--frame", required=True, help="Frame directory name inside each run")
    parser.add_argument("--run", action="append", required=True, help="Run directory (repeatable, left to right)")
    parser.add_argument("--label", action="append", required=True, help="Panel label (same order as --run)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise SystemExit("--run and --label counts must match")
    path = render_comparison(args.frame, args.run, args.label, args.output)
    print(path)


if __name__ == "__main__":
    main()
