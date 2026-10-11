"""Render object-track visualization for group-representative chains (read-only).

Consumes the temporal-groups chain evidence JSON (chain_basis=group_representatives)
plus the run's frame.jpg/masks.npz and draws one figure per video: a sampled-frame
grid where every object group representative carries its track tag (T##, stable
across frames; singletons get a thin contour only), a timeline strip whose bars
show each multi-frame track's per-second semantics (color changes expose flips),
and a black legend bar with counts. The figure supports human semantic review
only -- it never marks semantics verified.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 480
GAP = 12
COLS = 4
TIMELINE_ROW_H = 24
TIMELINE_PAD = 16
LEGEND_H = 118
FONT_CANDIDATES = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf")
FILL_ALPHA = 0.45
SAMPLING_INTERVAL = 2

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
BG = (24, 24, 24)
TEXT = (235, 235, 235)
MUTED = (170, 170, 170)


def load_font(size):
    for name in FONT_CANDIDATES:
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
    return ImageFont.load_default()


def semantic_color(semantic):
    return SEMANTIC_COLORS.get(semantic, DEFAULT_COLOR)


def load_payload(payload_path):
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    if payload.get("chain_basis") != "group_representatives":
        raise SystemExit("chain evidence must use chain_basis=group_representatives")
    frames = sorted(payload["per_frame"], key=lambda f: f["second"])
    multi = [c for c in payload["chains"] if c["length"] >= 2]
    singletons = [c for c in payload["chains"] if c["length"] == 1]
    tracks = {}
    for index, chain in enumerate(multi, start=1):
        tracks[chain["chain_id"]] = f"T{index:02d}"
    observations = {}
    for chain in payload["chains"]:
        tag = tracks.get(chain["chain_id"])
        for step in chain["steps"]:
            observations.setdefault(step["frame_id"], []).append((tag, step, chain))
    return payload, frames, multi, singletons, tracks, observations


def load_frame_layer(run_dir, frame_id, observations):
    frame_dir = Path(run_dir) / frame_id
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer((frame_dir / "frame.jpg").read_bytes(), np.uint8),
                                      cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    evidence = json.loads((frame_dir / "evidence.json").read_text(encoding="utf-8"))
    with np.load(frame_dir / "masks.npz") as archive:
        masks = {key: np.asarray(archive[key], dtype=bool) for key in archive.files}
    mask_by_id = {i["instance_id"]: masks[i["mask_key"]] for i in evidence["instances"]}
    layers = []
    for tag, step, chain in observations.get(frame_id, []):
        mask = mask_by_id[step["instance_id"]]
        layers.append({
            "tag": tag, "semantic": step["semantic"],
            "bbox": step["bbox_xyxy"], "mask": mask,
            "members": step.get("member_count", 1),
        })
    return image, layers


def render_panel(image, layers):
    overlay = image.astype(np.float64)
    for layer in layers:
        if layer["tag"] is not None:
            color = np.asarray(semantic_color(layer["semantic"]), dtype=np.float64)
            overlay[layer["mask"]] = (1 - FILL_ALPHA) * overlay[layer["mask"]] + FILL_ALPHA * color
    overlay = overlay.astype(np.uint8)
    for layer in layers:
        contours, _ = cv2.findContours(layer["mask"].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if layer["tag"] is not None:
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        else:
            cv2.drawContours(overlay, contours, -1, semantic_color(layer["semantic"]), 1)
    return overlay, image.shape[1]


def resize_width(image, width):
    scale = width / image.shape[1]
    return cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def draw_tag(draw, xy, text, font, scale):
    x = int(xy[0] * scale)
    y = int(xy[1] * scale)
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, fill=TEXT, font=font)


def render_timeline(draw, multi, tracks, x0, y0, width, seconds_axis, font, small):
    max_second = max(seconds_axis) + SAMPLING_INTERVAL
    plot_w = width - 110
    second_x = lambda s: x0 + 90 + (s / max_second) * plot_w
    for index, chain in enumerate(multi):
        y = y0 + index * TIMELINE_ROW_H
        tag = tracks[chain["chain_id"]]
        draw.text((x0, y), f"{tag}", fill=TEXT, font=small)
        for prev, step in zip([None] + chain["steps"], chain["steps"]):
            color = semantic_color(step["semantic"])
            x1 = second_x(step["second"])
            x2 = second_x(step["second"] + SAMPLING_INTERVAL)
            draw.rectangle((x1, y + 4, x2, y + 16), fill=color)
            if prev is not None and prev["semantic"] != step["semantic"]:
                draw.line((x1, y + 2, x1, y + 18), fill=(255, 255, 255), width=2)
        members = sum(s.get("member_count", 1) for s in chain["steps"])
        draw.text((x0 + width - 84, y), f"x{members}", fill=MUTED, font=small)
    axis_y = y0 + len(multi) * TIMELINE_ROW_H + 4
    for second in range(0, max_second + 1, SAMPLING_INTERVAL):
        x = second_x(second)
        draw.line((x, axis_y, x, axis_y + 5), fill=MUTED)
        label = f"{second}"
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((x - (box[2] - box[0]) // 2, axis_y + 7), label, fill=MUTED, font=small)
    return axis_y + 26


def render_track_figure(payload_path, run_dir, output, label):
    payload, frames, multi, singletons, tracks, observations = load_payload(payload_path)
    font = load_font(24)
    small = load_font(19)
    panels = []
    for frame in frames:
        image, layers = load_frame_layer(run_dir, frame["frame_id"], observations)
        panel, orig_w = render_panel(image, layers)
        scale = PANEL_W / orig_w
        panels.append((frame, resize_width(panel, PANEL_W), layers, scale))
    panel_h = panels[0][1].shape[0]
    rows = (len(panels) + COLS - 1) // COLS
    grid_w = COLS * PANEL_W + (COLS - 1) * GAP
    grid_h = rows * panel_h + (rows - 1) * GAP
    timeline_h = len(multi) * TIMELINE_ROW_H + 34
    canvas = Image.new("RGB", (grid_w, grid_h + TIMELINE_PAD + timeline_h + LEGEND_H), BG)
    draw = ImageDraw.Draw(canvas)
    for index, (frame, panel, layers, scale) in enumerate(panels):
        col, row = index % COLS, index // COLS
        x, y = col * (PANEL_W + GAP), row * (panel_h + GAP)
        canvas.paste(Image.fromarray(panel), (x, y))
        stamp = f"{frame['second']}s"
        box = draw.textbbox((x + 8, y + 6), stamp, font=small)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), stamp, fill=TEXT, font=small)
        for layer in layers:
            if layer["tag"] is not None:
                draw_tag(draw, (layer["bbox"][0], layer["bbox"][1]), layer["tag"], small, scale)
    y = grid_h + TIMELINE_PAD
    draw.text((0, y), "多帧物体轨线时间轴（分段颜色=该秒语义，白线=语义翻转，行尾 xN=吸收的原始实例数）",
              fill=MUTED, font=small)
    y = render_timeline(draw, multi, tracks, 0, y + 30, grid_w,
                        [f["second"] for f in frames], font, small)
    draw.rectangle((0, y, grid_w, y + LEGEND_H), fill=(16, 16, 16))
    draw.text((16, y + 10), label, fill=TEXT, font=font)
    flips = sum(1 for c in multi if c["flip_count"])
    instances = sum(sum(s.get("member_count", 1) for s in c["steps"]) for c in payload["chains"])
    draw.text((16, y + 44),
              f"物体轨线 {len(payload['chains'])}（多帧 {len(multi)} + 单例 {len(singletons)}）"
              f"  语义翻转链 {flips}  吸收原始实例 {instances}"
              f"  单例=仅一帧（细轮廓不填充）  T##=跨帧同一物体",
              fill=MUTED, font=small)
    semantics = sorted({s["semantic"] for c in payload["chains"] for s in c["steps"]})
    x = 16
    for semantic in semantics:
        color = semantic_color(semantic)
        draw.rectangle((x, y + 80, x + 18, y + 98), fill=color)
        draw.text((x + 24, y + 78), semantic, fill=MUTED, font=small)
        x += 24 + draw.textbbox((0, 0), semantic, font=small)[2] + 26
    output = Path(output)
    canvas.save(output, quality=92)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render object-track figure from group-representative chains (read-only)")
    parser.add_argument("--json", type=Path, required=True,
                        help="temporal groups chain evidence JSON (chain_basis=group_representatives)")
    parser.add_argument("--run", type=Path, required=True, help="Verified run directory with */frame.jpg and masks.npz")
    parser.add_argument("--label", required=True, help="Figure headline (video + pipeline identifier)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = render_track_figure(args.json, args.run, args.output, args.label)
    print(path)


if __name__ == "__main__":
    main()
