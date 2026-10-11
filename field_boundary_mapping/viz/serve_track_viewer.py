"""Read-only, loopback-only interactive viewer for group-representative object tracks.

Serves the temporal-groups chain evidence (chain_basis=group_representatives) for
one run: a large frame view where every object track is clickable (via its
representative bbox), a filmstrip of the sampled frames, a track list with
per-second semantic sparklines, and a detail card per track. Selecting a track
re-renders the frames with that object highlighted and everything else ghosted.

The viewer is derived evidence for human semantic review only: groups are review
statistics, not verified objects; nothing is edited, no source video is opened,
no inference runs, and semantic_verified stays false.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[2]   # 仓库根

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
FILL_ALPHA = 0.45
SELECT_ALPHA = 0.55
GHOST_CONTOUR = (90, 90, 90)
RENDER_CACHE_LIMIT = 320


def semantic_color(semantic: str) -> tuple[int, int, int]:
    return SEMANTIC_COLORS.get(semantic, DEFAULT_COLOR)


def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (color[2], color[1], color[0])


class TrackStore:
    """Loads the chain evidence JSON plus run frames and renders selections."""

    def __init__(self, evidence_path: Path | None, run_dir: Path,
                 boundaries_path: Path | None = None):
        self.run_dir = run_dir
        self._render_cache: dict[tuple[str, str], bytes] = {}
        self._verified_sources: dict[str, dict[str, bytes]] | None = None
        self._raw_content_types: dict[str, str] = {}
        self.visible_boundaries = None
        self.payload: dict | None = None
        self.frames: list[dict] | None = None
        self.frame_ids: set[str] | None = None
        self.tracks: list[dict] = []
        self.tracks_by_id: dict[str, dict] = {}
        self.observations: dict[str, list[dict]] = {}
        if evidence_path is not None:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            if payload.get("chain_basis") != "group_representatives":
                raise SystemExit("chain evidence must use chain_basis=group_representatives")
            self.payload = payload
            frames = sorted(payload["per_frame"], key=lambda f: f["second"])
            self.frames = [{"frame_id": f["frame_id"], "second": f["second"]} for f in frames]
            self.frame_ids = {f["frame_id"] for f in self.frames}

            multi = [c for c in payload["chains"] if c["length"] >= 2]
            singletons = [c for c in payload["chains"] if c["length"] == 1]
            multi.sort(key=lambda c: (-c["length"], c["seconds"][0]))
            singletons.sort(key=lambda c: c["seconds"][0])
            for index, chain in enumerate(multi, start=1):
                self._add_track(chain, f"T{index:02d}")
            for index, chain in enumerate(singletons, start=1):
                self._add_track(chain, f"S{index:02d}")
            self.tracks_by_id = {t["chain_id"]: t for t in self.tracks}
            for track in self.tracks:
                for step in track["steps"]:
                    self.observations.setdefault(step["frame_id"], []).append(
                        {"chain_id": track["chain_id"], "tag": track["tag"],
                         "singleton": track["singleton"], "step": step})
        elif boundaries_path is None:
            raise ValueError("either chain evidence JSON or boundaries JSON is required")
        if boundaries_path is not None:
            # Keep validation optional; pin the actual bytes used by every render.
            from field_boundary_mapping.evidence.extract_visible_boundaries import contained_file, load_boundary_payload, source_manifest

            boundaries = load_boundary_payload(boundaries_path, run_dir)
            boundary_frames = boundaries["frames"]
            boundary_ids = {frame["frame_id"] for frame in boundary_frames}
            if self.frame_ids is None:
                # Boundary-only mode: sub-second runs carry no chain evidence yet, so
                # the frame list itself comes from the boundary payload's real times.
                if len(boundary_frames) != len(boundary_ids):
                    raise ValueError("boundary payload contains duplicate frame_ids")
                self.frames = [{"frame_id": frame["frame_id"], "second": frame["capture_time_s"]}
                               for frame in sorted(boundary_frames, key=lambda f: f["capture_time_s"])]
                self.frame_ids = boundary_ids
                self.payload = {"run": boundaries["run"], "chain_basis": "boundary_only_no_temporal_chains",
                                "iou_threshold": None, "sampled_seconds": None}
            elif boundary_ids != self.frame_ids or len(boundary_frames) != len(self.frame_ids):
                raise ValueError("boundary frame_id set must exactly match chain frames (no duplicates)")
            root, manifest, manifest_hash = source_manifest(run_dir)
            if manifest_hash != boundaries["source_manifest_sha256"]:
                raise ValueError("Boundary evidence does not match source run")
            entries = {entry["frame_id"]: entry for entry in manifest["frames"]}
            if set(entries) != boundary_ids:
                raise ValueError("boundary frame_id set must exactly match source manifest")
            self._verified_sources = {}
            for frame in boundary_frames:
                frame_id = frame["frame_id"]
                entry = entries[frame_id]
                sources = {}
                for field in ("image_path", "evidence_path", "masks_path"):
                    relative = entry[field]
                    data = contained_file(root, relative).read_bytes()
                    # Hash the bytes we retain, not a separate read of the file.
                    if hashlib.sha256(data).hexdigest() != entry["artifact_sha256"].get(relative):
                        raise ValueError(f"Source artifact hash mismatch: {relative}")
                    sources[field] = data
                self._verified_sources[frame_id] = sources
                image = self._read_frame(frame_id)
                width, height = frame["image"]["width"], frame["image"]["height"]
                if (type(width) is not int or type(height) is not int
                        or (width, height) != (image.shape[1], image.shape[0])):
                    raise ValueError(f"boundary image dimensions mismatch for {frame_id}")
                raw = sources["image_path"]
                if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                    self._raw_content_types[frame_id] = "image/png"
                elif raw.startswith(b"\xff\xd8\xff"):
                    self._raw_content_types[frame_id] = "image/jpeg"
                else:
                    raise ValueError("Boundary source image must be JPEG or PNG")
            self.visible_boundaries = boundaries
        first = self._read_frame(self.frames[0]["frame_id"])
        self.frame_size = (first.shape[1], first.shape[0])

    def _add_track(self, chain: dict, tag: str) -> None:
        members = sum(s.get("member_count", 1) for s in chain["steps"])
        self.tracks.append({
            "chain_id": chain["chain_id"], "tag": tag,
            "singleton": chain["length"] == 1, "length": chain["length"],
            "seconds": chain["seconds"], "semantic_sequence": chain["semantic_sequence"],
            "flip_count": chain["flip_count"], "modal_semantic": chain["modal_semantic"],
            "modal_fraction": chain["modal_fraction"],
            "appearance_rate": chain["appearance_rate"], "members": members,
            "steps": chain["steps"],
        })

    def _frame_bytes(self, frame_id: str) -> bytes:
        if self._verified_sources is not None:
            return self._verified_sources[frame_id]["image_path"]
        return (self.run_dir / frame_id / "frame.jpg").read_bytes()

    def _read_frame(self, frame_id: str) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(self._frame_bytes(frame_id), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable frame {frame_id}")
        return image

    def _layer_mask(self, frame_id: str, step: dict) -> np.ndarray:
        if self._verified_sources is None:
            evidence = json.loads((self.run_dir / frame_id / "evidence.json").read_text(encoding="utf-8"))
            masks = self.run_dir / frame_id / "masks.npz"
        else:
            sources = self._verified_sources[frame_id]
            evidence = json.loads(sources["evidence_path"])
            masks = io.BytesIO(sources["masks_path"])
        with np.load(masks, allow_pickle=False) as archive:
            key = next(i["mask_key"] for i in evidence["instances"] if i["instance_id"] == step["instance_id"])
            return np.asarray(archive[key], dtype=bool)

    def render_content_type(self, frame_id: str, selection: str) -> str:
        return self._raw_content_types.get(frame_id, "image/jpeg") if selection == "raw" else "image/jpeg"

    def render(self, frame_id: str, selection: str) -> bytes:
        """Return exact source bytes for "raw", otherwise a JPEG track rendering."""
        if selection == "raw":
            # Preserve source pixels: no masks, annotations or re-encoding.
            return self._frame_bytes(frame_id)
        key = (frame_id, selection)
        cached = self._render_cache.get(key)
        if cached is not None:
            return cached
        image = self._read_frame(frame_id)
        layers = self.observations.get(frame_id, [])
        overlays = []
        for layer in layers:
            step = layer["step"]
            mask = self._layer_mask(frame_id, step)
            overlays.append((layer, mask))
        selected = [o for o in overlays if o[0]["chain_id"] == selection]
        ghosts = [o for o in overlays if o[0]["chain_id"] != selection]
        if selection == "all":
            selected, ghosts = overlays, []
        blend = image.astype(np.float64)
        for layer, mask in selected:
            color = np.asarray(rgb_to_bgr(semantic_color(layer["step"]["semantic"])), dtype=np.float64)
            alpha = SELECT_ALPHA if selection != "all" else FILL_ALPHA
            blend[mask] = (1 - alpha) * blend[mask] + alpha * color
        image = blend.astype(np.uint8)
        for layer, mask in ghosts:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, GHOST_CONTOUR, 1)
        for layer, mask in selected:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            color = rgb_to_bgr(semantic_color(layer["step"]["semantic"]))
            thickness = 3 if selection != "all" else 2
            if layer["singleton"] and selection == "all":
                cv2.drawContours(image, contours, -1, color, 1)
                continue
            cv2.drawContours(image, contours, -1, (255, 255, 255), thickness)
            if (not layer["singleton"] or selection == layer["chain_id"]) and mask.any():
                # Zero padding also handles masks touching every image edge.
                distance = cv2.distanceTransform(
                    np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
                y, x = (int(v) for v in np.unravel_index(distance.argmax(), mask.shape))
                label = layer["tag"]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # Keep the text anchor at that mask pixel, even for small masks.
                cv2.rectangle(image, (x - 4, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("jpeg encode failed")
        data = buffer.tobytes()
        if len(self._render_cache) >= RENDER_CACHE_LIMIT:
            self._render_cache.pop(next(iter(self._render_cache)))
        self._render_cache[key] = data
        return data

    def api_data(self) -> dict:
        colors = {name: "#%02x%02x%02x" % color for name, color in SEMANTIC_COLORS.items()}
        tracks = []
        for track in self.tracks:
            entry = {k: v for k, v in track.items() if k != "steps"}
            entry["steps"] = [
                {"frame_id": s["frame_id"], "second": s["second"], "semantic": s["semantic"],
                 "score": s["score"], "bbox_xyxy": s["bbox_xyxy"],
                 "group_id": s.get("group_id"), "member_count": s.get("member_count", 1),
                 "passed_count": s.get("passed_count")}
                for s in track["steps"]
            ]
            tracks.append(entry)
        return {
            "run": self.payload["run"], "chain_basis": self.payload["chain_basis"],
            "iou_threshold": self.payload["iou_threshold"],
            "sampled_seconds": self.payload["sampled_seconds"],
            "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "frames": self.frames, "tracks": tracks, "colors": colors,
            "semantic_verified": False, "annotation_used_for_inference": False,
            "note": ("组是派生审阅统计，不是已核验物体；单例=仅一帧。"
                     "本页只读，支持人工语义核验，不修改任何数据。"),
            "visible_boundaries": self.visible_boundaries,
        }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>农田边界 · 物体轨线查看器</title>
<style nonce="__NONCE__">
:root{color-scheme:light;--ink:#20342d;--muted:#63746c;--line:#dbe4dd;--green:#256448;--amber:#8a6d1f}
*{box-sizing:border-box}body{margin:0;background:#f3f6f2;color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{padding:22px 32px;background:#fff;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px;align-items:center}
h1{font-size:22px;margin:0}p{margin:4px 0 0}.subtle{color:var(--muted);font-size:13px}.badge{border:1px solid #bbd6c5;color:var(--green);border-radius:20px;padding:5px 14px;white-space:nowrap}
main{max-width:1680px;margin:auto;padding:20px 32px}
.warning{padding:12px 18px;border:1px solid #e5c87e;background:#fff6dc;border-radius:9px;color:#6c5015;margin-bottom:16px;font-size:14px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
button,select{font:inherit;border:1px solid #c9d5cc;border-radius:7px;background:#fff;color:var(--ink);padding:7px 13px}button{cursor:pointer}button:hover{border-color:var(--green)}button[aria-pressed=true]{background:var(--green);border-color:var(--green);color:#fff}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:20px;align-items:start}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}.card-body{padding:16px}
.viewport{position:relative;background:#18251f}.viewport img{width:100%;display:block}
.hitbox{position:absolute;border:0;background:transparent;cursor:pointer}
.hitbox:hover{box-shadow:0 0 0 3px #ffe57a inset;background:rgba(255,229,122,.18)}
.hitbox.picked{box-shadow:0 0 0 3px #ffe57a inset}
.caption{padding:10px 16px;font-size:13px;color:var(--muted);border-top:1px solid var(--line)}
.filmstrip{display:flex;gap:8px;overflow-x:auto;padding:12px 14px;border-top:1px solid var(--line);background:#fafcf9}
.thumb{flex:0 0 150px;border:2px solid transparent;border-radius:6px;overflow:hidden;padding:0;background:none;position:relative}
.thumb img{width:100%;display:block}
.thumb .sec{position:absolute;left:4px;top:4px;background:rgba(0,0,0,.75);color:#fff;font-size:12px;padding:1px 6px;border-radius:4px}
.thumb.active{border-color:var(--green)}
.thumb.absent img{filter:grayscale(.6) brightness(.75)}
.tracklist{max-height:430px;overflow:auto}.trackrow{display:flex;align-items:center;gap:10px;width:100%;padding:8px 10px;border:0;border-bottom:1px solid #eef3ef;background:#fff;text-align:left}
.trackrow:hover{background:#f4f8f5}.trackrow.active{background:#e7f1ea;border-color:#bbd6c5}
.tag{font:700 13px/1 ui-monospace,monospace;background:#20342d;color:#fff;border-radius:5px;padding:4px 7px;min-width:38px;text-align:center}
.spark{flex:1;display:flex;height:14px;border-radius:3px;overflow:hidden}
.spark i{flex:1}.spark i.flip{box-shadow:-2px 0 0 #fff inset}
.tinfo{font:12px/1.3 ui-monospace,monospace;color:var(--muted);white-space:nowrap}
.details{font-size:13px}.details dt{color:var(--muted);font-size:12px;margin-top:8px}.details dd{margin:2px 0}
table{border-collapse:collapse;width:100%;font:12px/1.5 ui-monospace,monospace}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid #eef3ef}
.chip{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.flip-row td{background:#fff6dc}
h2{font-size:16px;margin:0 0 10px}
footer{padding:18px 0 26px;font-size:12px;color:var(--muted)}
[hidden]{display:none!important}
#boundaryoverlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.boundaryline{fill:none;stroke:#27cbe0;stroke-width:3;vector-effect:non-scaling-stroke;pointer-events:stroke;cursor:pointer}
.boundaryline.coverageline{stroke:#ffce40;stroke-width:2;stroke-dasharray:9 3}
.boundaryline.gapclosedline{stroke:#8ce06b;stroke-width:2;stroke-dasharray:2 5;stroke-linecap:round}
.boundaryline:hover,.boundaryline:focus,.boundaryline.picked{stroke:#ff6538;stroke-width:5;outline:none}
.diagnosticline{fill:none;stroke-width:1.5;stroke-dasharray:5 4;vector-effect:non-scaling-stroke;pointer-events:none}
.reason-unknown{color:#ddd;stroke:#ddd}.reason-conflict{color:#e56df0;stroke:#e56df0}
.reason-occlusion{color:#f0a72b;stroke:#f0a72b}.reason-fov{color:#a8b5c5;stroke:#a8b5c5}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}
.legend i{display:inline-block;width:20px;border-top:2px dashed currentColor;vertical-align:middle;margin-right:4px}
.boundaryrow{flex-wrap:wrap}.boundaryrow .tinfo{margin-left:auto}
#boundarydetails dd{overflow-wrap:anywhere}
@media(max-width:1000px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div><h1>农田边界 · 物体轨线查看器</h1><p class="subtle" id="subtitle">读取链证据中…</p></div>
  <span class="badge">127.0.0.1 · 只读</span>
</header>
<main>
  <div class="warning"><strong>组是派生审阅统计，不是已核验物体。</strong>选择轨线只改变显示高亮，不修改、不删除任何候选；语义冲突只标记不裁决。</div>
  <div id="boundarynotice" class="warning" hidden><strong>冠层/沟渠轮廓≠物理田界，裸地≠安全掉头区</strong>。B 为双侧交界候选（image_candidate），C 为单侧覆盖候选（one_sided_candidate），GC 为闭合候选（gap_closed_candidate）：两侧掩码从未直接接触，由窄 unknown 带按就近分配闭合，是假设几何而非观测证据，永不升级 B；语义、物理田界与机动可行性均未核验，不使用标注推理。ground 包含 ground 与 harvested_ground，不等于必然裸土；来源细类在详情中保留。B/C/GC 编号仅在本帧有效，不是跨帧实体身份。</div>
  <div class="toolbar">
    <label id="modecontrol" hidden>模式 <select id="viewmode" aria-label="查看模式"><option value="boundaries">边界候选</option><option value="raw">原始图像</option><option value="tracks">候选轨线</option></select></label>
    <button id="prev" type="button">← 上一帧</button>
    <select id="frameselect" aria-label="选择帧"></select>
    <button id="next" type="button">下一帧 →</button>
    <button id="showall" type="button" aria-pressed="false">显示全部轨线</button>
    <span id="hint" class="subtle">←/→ 切帧 · Esc 回到全部 · 点击画面实体选择该物体轨线</span>
  </div>
  <div id="diagnosticcontrols" class="toolbar" hidden>
    <label><input id="showcoverage" type="checkbox" checked>显示单侧覆盖轮廓 C（另一侧未知）</label>
    <label><input id="showgapclosed" type="checkbox">显示闭合候选 GC（假设几何，非观测接触）</label>
    <label><input id="showdiagnostics" type="checkbox">显示未知/冲突/遮挡/视野截断</label>
    <div class="legend" aria-label="诊断线图例">
      <span><i class="reason-unknown"></i>未知 unknown</span><span><i class="reason-conflict"></i>冲突 conflict</span>
      <span><i class="reason-occlusion"></i>遮挡 occlusion</span><span><i class="reason-fov"></i>视野截断 fov</span>
    </div>
    <span class="subtle">青实线 B：两侧支持；黄虚线 C：单侧覆盖轮廓，可能包含冠层对天空或遮挡间隙，不是已确认地表交界；绿点线 GC：窄 unknown 带按就近闭合的假设几何，不是观测交界，永不升级 B；橙红：已选。诊断线不计入交界。</span>
  </div>
  <div class="layout">
    <section class="card">
      <div class="viewport"><img id="mainimage" alt="当前帧叠加图"><div id="hitboxes"></div><svg id="boundaryoverlay" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" aria-label="单帧边界候选" hidden></svg></div>
      <div class="caption" id="maincaption"></div>
      <div class="filmstrip" id="filmstrip"></div>
    </section>
    <aside>
      <section id="trackpanel" class="card card-body"><h2>物体轨线</h2><div id="trackcount" class="subtle"></div><div class="tracklist" id="tracklist"></div></section>
      <section id="trackdetailpanel" class="card card-body"><h2>轨线详情</h2><div class="details" id="details"><span class="subtle">点击轨线或画面实体查看详情。</span></div></section>
      <section id="boundarypanel" class="card card-body" hidden><h2>本帧边界候选</h2><p class="subtle">B/C/GC 编号仅帧内有效，不表示实体身份。</p><div id="boundarylist" class="tracklist"></div></section>
      <section id="boundarydetailpanel" class="card card-body" hidden><h2>边界候选详情</h2><div id="boundarydetails" class="details"></div></section>
      <section id="rawpanel" class="card card-body" hidden><h2>原始图像</h2><p class="subtle">源帧图像，不叠加掩码、T 框或候选线。</p></section>
    </aside>
  </div>
  <footer>物体轨线查看器 · 证据支持人工语义核验，semantic_verified 恒为 false · 不打开源视频、不执行推理。</footer>
</main>
<script nonce="__NONCE__">
let DATA = null;
let state = { frame: 0, selection: "all", mode: "tracks", boundary: null };

const $ = (id) => document.getElementById(id);

function colorOf(semantic) {
  return (DATA.colors && DATA.colors[semantic]) || "#787878";
}

function frameList() {
  return DATA.frames;
}

function currentFrame() {
  return frameList()[state.frame];
}

function boundaryFrame(frame = currentFrame()) {
  return DATA.visible_boundaries?.frames.find(f => f.frame_id === frame.frame_id);
}

function frameTimeLabel(frame) {
  const boundary = boundaryFrame(frame);
  if (!boundary) return frame.second + "s";
  if (String(frame.second) === String(boundary.capture_time_s)) {
    return "实际采集 " + boundary.capture_time_s + "s";
  }
  return "采样 " + frame.second + "s · 实际采集 " + boundary.capture_time_s + "s";
}

function setFrame(index) {
  state.frame = (index + frameList().length) % frameList().length;
  state.boundary = null; // B identifiers are frame-local, never a track identity.
  refresh();
}

function renderSelectionValue() {
  return state.mode === "tracks" ? state.selection : "raw";
}

function imgSrc(frame) {
  return "/render/" + encodeURIComponent(frame.frame_id) + "/" + encodeURIComponent(renderSelectionValue()) + ".jpg";
}

function selectTrack(chainId) {
  state.selection = chainId;
  if (state.selection !== "all") {
    const track = DATA.tracks.find(t => t.chain_id === chainId);
    const idx = frameList().findIndex(f => f.frame_id === track.steps[0].frame_id);
    if (idx >= 0) state.frame = idx;
  }
  refresh();
}

function setAll() {
  state.selection = "all";
  refresh();
}

function refresh() {
  const frame = currentFrame();
  const tracksMode = state.mode === "tracks";
  const boundariesMode = state.mode === "boundaries";
  $("mainimage").src = imgSrc(frame);
  $("mainimage").alt = tracksMode ? "当前帧轨线叠加图" : "当前帧原始图像";
  $("showall").hidden = !tracksMode;
  $("trackpanel").hidden = !tracksMode;
  $("trackdetailpanel").hidden = !tracksMode;
  $("boundarypanel").hidden = !boundariesMode;
  $("boundarydetailpanel").hidden = !boundariesMode;
  $("diagnosticcontrols").hidden = !boundariesMode;
  $("rawpanel").hidden = state.mode !== "raw";
  $("hint").textContent = tracksMode ? "←/→ 切帧 · Esc 回到全部 · 点击画面实体选择该物体轨线" :
    boundariesMode ? "←/→ 切帧 · 点击候选线或列表选择本帧 B/C/GC 编号 · Esc 清空选择" : "←/→ 切帧 · 原始图像无叠加";
  const present = DATA.tracks.filter(t => t.steps.some(s => s.frame_id === frame.frame_id));
  const box = $("hitboxes");
  box.replaceChildren();
  for (const track of tracksMode ? present : []) {
    const step = track.steps.find(s => s.frame_id === frame.frame_id);
    const [x0, y0, x1, y1] = step.bbox_xyxy;
    const div = document.createElement("button");
    div.type = "button";
    div.className = "hitbox" + (track.chain_id === state.selection ? " picked" : "");
    div.title = track.tag + " · " + step.semantic + " · score " + step.score;
    div.style.left = (x0 / DATA.frame_size.width * 100) + "%";
    div.style.top = (y0 / DATA.frame_size.height * 100) + "%";
    div.style.width = ((x1 - x0) / DATA.frame_size.width * 100) + "%";
    div.style.height = ((y1 - y0) / DATA.frame_size.height * 100) + "%";
    div.addEventListener("click", () => selectTrack(track.chain_id));
    box.appendChild(div);
  }
  const selectedTrack = DATA.tracks.find(t => t.chain_id === state.selection);
  const missing = selectedTrack && !present.some(t => t.chain_id === selectedTrack.chain_id);
  let caption = frameTimeLabel(frame) + " · 帧 " + (state.frame + 1) + "/" + frameList().length;
  if (tracksMode) {
    caption += " · 当前帧实体 " + present.length + "（多帧轨线 " + present.filter(t => !t.singleton).length + " + 单例 " + present.filter(t => t.singleton).length + "）" +
      (state.selection === "all" ? "" : " · 已选 " + selectedTrack.tag + (missing ? "（该物体在此帧无观测，画面仅剩其余实体暗轮廓）" : ""));
  } else if (boundariesMode) {
    const gcCount = (boundaryFrame().gap_closed_boundaries || []).length;
    caption += " · 双侧交界 B " + boundaryFrame().boundaries.length + " · 单侧轮廓 C " + boundaryFrame().coverage_contours.length +
      (gcCount ? " · 闭合候选 GC " + gcCount + "（假设几何）" : "") + " · 编号仅帧内有效";
  } else {
    caption += " · 原始图像（无叠加）";
  }
  $("maincaption").textContent = caption;
  $("frameselect").value = String(state.frame);
  $("showall").setAttribute("aria-pressed", state.selection === "all");
  renderFilmstrip();
  renderBoundaries();
  if (tracksMode) {
    renderTrackList();
    renderDetails();
  }
}

function semanticLabel(semantic) {
  const labels = {crop: "作物", ground: "地表", woody_vegetation: "木本植被", water: "水体", ditch: "沟渠", unknown: "未知"};
  return (labels[semantic] || semantic) + " (" + semantic + ")";
}

function selectBoundary(boundaryId) {
  state.boundary = boundaryId;
  refresh();
}

function boundaryPolyline(points, className) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", points.map(point => point.join(",")).join(" "));
  line.setAttribute("class", className);
  return line;
}

function renderBoundaries() {
  const svg = $("boundaryoverlay");
  const list = $("boundarylist");
  svg.replaceChildren();
  list.replaceChildren();
  $("boundarydetails").replaceChildren();
  // SVG uses an attribute rather than HTMLElement.hidden in some browsers.
  svg.toggleAttribute("hidden", state.mode !== "boundaries");
  if (state.mode !== "boundaries") return;
  const frame = boundaryFrame();
  // Integer coordinates denote pixel centers; retain half-pixel contour points.
  svg.setAttribute("viewBox", "-0.5 -0.5 " + frame.image.width + " " + frame.image.height);
  if ($("showdiagnostics").checked) {
    for (const diagnostic of frame.diagnostics) {
      for (const points of diagnostic.polylines_px) {
        svg.appendChild(boundaryPolyline(points, "diagnosticline reason-" + diagnostic.reason));
      }
    }
  }
  if (!frame.boundaries.length) {
    const empty = document.createElement("p");
    empty.textContent = "无两侧支持的交界候选，不代表无边界或可通行";
    list.appendChild(empty);
  }
  const candidates = [...frame.boundaries,
    ...($("showcoverage").checked ? frame.coverage_contours : []),
    ...($("showgapclosed").checked ? (frame.gap_closed_boundaries || []) : [])];
  for (const boundary of candidates) {
    const picked = boundary.boundary_id === state.boundary;
    const oneSided = boundary.status === "one_sided_candidate";
    const gapClosed = boundary.status === "gap_closed_candidate";
    const label = boundary.boundary_id + " · " + semanticLabel(boundary.side_a.semantic) + " ↔ " + semanticLabel(boundary.side_b.semantic) + (gapClosed ? " · 假设几何" : "");
    const line = boundaryPolyline(boundary.points_px, "boundaryline" + (oneSided ? " coverageline" : "") + (gapClosed ? " gapclosedline" : "") + (picked ? " picked" : ""));
    line.setAttribute("tabindex", "0");
    line.setAttribute("role", "button");
    line.setAttribute("aria-label", label);
    line.setAttribute("aria-pressed", String(picked));
    line.addEventListener("click", () => selectBoundary(boundary.boundary_id));
    line.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectBoundary(boundary.boundary_id);
      }
    });
    svg.appendChild(line);
    const row = document.createElement("button");
    row.type = "button";
    row.className = "trackrow boundaryrow" + (picked ? " active" : "");
    row.setAttribute("aria-pressed", String(picked));
    const name = document.createElement("span");
    name.textContent = label;
    const length = document.createElement("span");
    length.className = "tinfo";
    length.textContent = boundary.length_px.toFixed(1) + " px";
    row.append(name, length);
    row.addEventListener("click", () => selectBoundary(boundary.boundary_id));
    list.appendChild(row);
  }
  renderBoundaryDetails(frame);
}

function renderBoundaryDetails(frame) {
  const box = $("boundarydetails");
  const boundary = [...frame.boundaries, ...frame.coverage_contours, ...(frame.gap_closed_boundaries || [])].find(b => b.boundary_id === state.boundary);
  if (!boundary) {
    box.textContent = "点击本帧 B/C 候选查看来源与端点原因；C 仅有单侧支持，另一侧未知；GC 为闭合假设几何，用上方开关开启。切帧会清空选择。";
    return;
  }
  const details = document.createElement("dl");
  function item(name, value) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = name;
    dd.textContent = value;
    details.append(dt, dd);
  }
  item("本帧编号（非实体身份）", boundary.boundary_id);
  item("采样与实际时间", "采样 " + currentFrame().second + "s · capture_time_s=" + frame.capture_time_s + "s（实际采集）");
  item("类型 / 状态", boundary.type + " / " + boundary.status);
  item("证据层级", boundary.status === "one_sided_candidate" ? "仅单侧覆盖轮廓，另一侧未识别；不是已确认地表交界。"
    : boundary.status === "gap_closed_candidate" ? "假设几何：两侧掩码从未直接接触，中间 unknown 窄带按就近分配闭合；非观测证据，永不升级 B。" : "同帧两侧掩码直接接触；语义仍未核验。");
  item("像素长度 / 闭合", boundary.length_px.toFixed(1) + " px / " + (boundary.closed ? "掩码轮廓闭合，不是田块闭合" : "否"));
  if (boundary.gap) {
    item("闭合带宽 width_px（min/mean/max）", boundary.gap.width_px.min.toFixed(1) + " / " + boundary.gap.width_px.mean.toFixed(1) + " / " + boundary.gap.width_px.max.toFixed(1) + " px");
    item("闭合重分配像素", "assigned_neighbor_px=" + boundary.gap.assigned_neighbor_px + " · A 侧 " + boundary.gap.side_a_assigned_px + " px · B 侧 " + boundary.gap.side_b_assigned_px + " px");
  }
  for (const [name, side] of [["A 侧", boundary.side_a], ["B 侧", boundary.side_b]]) {
    item(name + "类别", semanticLabel(side.semantic));
    item(name + "来源 instance_ids", side.instance_ids.join(", "));
    item(name + "来源细类 source_semantics", JSON.stringify(side.source_semantics));
  }
  boundary.endpoint_reasons.forEach((reasons, index) => item("端点 " + (index + 1) + " · endpoint_reasons", reasons.join(", ")));
  item("核验状态", "semantic_verified=false · physical_field_boundary_verified=false · maneuver_verified=false");
  box.appendChild(details);
}

function renderFilmstrip() {
  const strip = $("filmstrip");
  strip.innerHTML = "";
  const selectedTrack = state.mode === "tracks" && DATA.tracks.find(t => t.chain_id === state.selection);
  for (let i = 0; i < frameList().length; i++) {
    const frame = frameList()[i];
    const btn = document.createElement("button");
    btn.type = "button";
    const absent = selectedTrack && !selectedTrack.steps.some(s => s.frame_id === frame.frame_id);
    btn.className = "thumb" + (i === state.frame ? " active" : "") + (absent ? " absent" : "");
    const img = document.createElement("img");
    img.src = "/render/" + encodeURIComponent(frame.frame_id) + (state.mode === "tracks" ? "/all.jpg" : "/raw.jpg");
    img.alt = frameTimeLabel(frame);
    const sec = document.createElement("span");
    sec.className = "sec";
    sec.textContent = frameTimeLabel(frame);
    btn.appendChild(img);
    btn.appendChild(sec);
    btn.addEventListener("click", () => setFrame(i));
    strip.appendChild(btn);
  }
}

function renderTrackList() {
  const list = $("tracklist");
  list.innerHTML = "";
  const multi = DATA.tracks.filter(t => !t.singleton);
  const single = DATA.tracks.filter(t => t.singleton);
  $("trackcount").textContent = "物体轨线 " + DATA.tracks.length + "（多帧 " + multi.length + " + 单例 " + single.length + "）";
  for (const track of multi) {
    list.appendChild(trackRow(track));
  }
  const head = document.createElement("div");
  head.className = "trackrow";
  head.style.fontWeight = "600";
  head.textContent = "单例（仅出现一帧，细轮廓）";
  list.appendChild(head);
  for (const track of single) {
    list.appendChild(trackRow(track));
  }
}

function trackRow(track) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "trackrow" + (track.chain_id === state.selection ? " active" : "");
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = track.tag;
  const spark = document.createElement("span");
  spark.className = "spark";
  track.steps.forEach((step, i) => {
    const seg = document.createElement("i");
    seg.style.background = colorOf(step.semantic);
    if (i > 0 && track.steps[i - 1].semantic !== step.semantic) seg.className = "flip";
    spark.appendChild(seg);
  });
  const info = document.createElement("span");
  info.className = "tinfo";
  info.textContent = track.length + "帧 x" + track.members;
  row.appendChild(tag);
  row.appendChild(spark);
  row.appendChild(info);
  row.addEventListener("click", () => selectTrack(track.chain_id));
  return row;
}

function renderDetails() {
  const box = $("details");
  box.replaceChildren();
  const track = DATA.tracks.find(t => t.chain_id === state.selection);
  if (!track) {
    box.textContent = "点击轨线或画面实体查看详情。";
    return;
  }
  const details = document.createElement("dl");
  const fields = [
    ["候选轨线（不是已核验物体）", track.tag + " · " + track.length + " 帧（" + track.seconds[0] + "–" + track.seconds.at(-1) + "s）"],
    ["模态语义（汇总，非逐帧真值）", track.modal_semantic + "（" + Math.round(track.modal_fraction * 100) + "%） · 语义翻转 " + track.flip_count + " 次"],
    ["吸收原始实例", track.members + "（组代表=组内最高分通过实例）"]
  ];
  for (const [name, value] of fields) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = name;
    dd.textContent = value;
    details.append(dt, dd);
  }
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const name of ["秒", "语义", "分数", "组", "成员"]) {
    const th = document.createElement("th");
    th.textContent = name;
    head.appendChild(th);
  }
  const body = table.createTBody();
  track.steps.forEach((step, index) => {
    const row = body.insertRow();
    if (index && track.steps[index - 1].semantic !== step.semantic) row.className = "flip-row";
    for (const value of [step.second + "s", step.semantic, step.score.toFixed(3), step.group_id || "-", step.member_count]) {
      row.insertCell().textContent = String(value);
    }
    const chip = document.createElement("i");
    chip.className = "chip";
    chip.style.backgroundColor = colorOf(step.semantic);
    row.cells[1].prepend(chip);
  });
  box.append(details, table);
}

async function boot() {
  const response = await fetch("/api/data");
  DATA = await response.json();
  document.title = "物体轨线查看器 · " + DATA.run;
  $("subtitle").textContent = "run=" + DATA.run + " · chain_basis=" + DATA.chain_basis +
    (DATA.iou_threshold === null ? "" : " · IoU 阈值 " + DATA.iou_threshold) +
    " · 采样 " + DATA.frames.length + " 帧（" + DATA.note + "）";
  const hasBoundaries = DATA.visible_boundaries !== null;
  $("modecontrol").hidden = !hasBoundaries;
  $("boundarynotice").hidden = !hasBoundaries;
  if (!DATA.tracks.length) {
    const option = $("viewmode").querySelector('option[value="tracks"]');
    if (option) option.remove();
  }
  state.mode = hasBoundaries ? "boundaries" : "tracks";
  $("viewmode").value = state.mode;
  $("viewmode").addEventListener("change", () => {
    state.mode = $("viewmode").value;
    state.boundary = null;
    refresh();
  });
  $("showdiagnostics").addEventListener("change", refresh);
  $("showcoverage").addEventListener("change", () => { state.boundary = null; refresh(); });
  $("showgapclosed").addEventListener("change", () => { state.boundary = null; refresh(); });
  const select = $("frameselect");
  frameList().forEach((frame, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = frameTimeLabel(frame);
    select.appendChild(option);
  });
  select.addEventListener("change", () => setFrame(Number(select.value)));
  $("prev").addEventListener("click", () => setFrame(state.frame - 1));
  $("next").addEventListener("click", () => setFrame(state.frame + 1));
  $("showall").addEventListener("click", setAll);
  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      setFrame(state.frame + (event.key === "ArrowLeft" ? -1 : 1));
    } else if (event.key === "Escape") {
      if (state.mode === "tracks") setAll();
      else { state.boundary = null; refresh(); }
    }
  });
  refresh();
}

boot().catch((error) => {
  document.getElementById("subtitle").textContent = "加载失败：" + error.message;
});
</script>
</body>
</html>
'''


def create_server(evidence_path: str | Path | None, run_dir: str | Path, port: int = 8768,
                  label: str | None = None, boundaries: Path | None = None) -> ThreadingHTTPServer:
    """Build a read-only server; port=0 is useful for isolated tests."""
    store = TrackStore(None if evidence_path is None else Path(evidence_path),
                       Path(run_dir), boundaries_path=boundaries)
    nonce = secrets.token_urlsafe(24)
    page = HTML.replace("__NONCE__", nonce).encode("utf-8")
    api_payload = json.dumps(store.api_data(), ensure_ascii=False, allow_nan=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "FieldTrackViewer"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, data: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", (
                "default-src 'none'; img-src 'self'; connect-src 'self'; "
                f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            ))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if self.command != "HEAD":
                self.wfile.write(data)

        def send_error(self, code: int, message: str | None = None,
                       explain: str | None = None) -> None:
            self._send(code, json.dumps({"error": message or "请求未被允许"},
                                        ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:
            host_port = self.server.server_address[1]
            if self.headers.get("Host", "").lower() not in {
                f"127.0.0.1:{host_port}", f"localhost:{host_port}",
            }:
                self.send_error(403, "仅接受本机地址访问")
                return
            try:
                raw = self.path
                if not raw.startswith("/") or raw.startswith("//") or any(ord(c) < 32 or ord(c) == 127 for c in raw):
                    raise ValueError("非法请求路径")
                from urllib.parse import urlsplit, unquote
                url = urlsplit(raw)
                if url.query or url.fragment or url.netloc or url.scheme:
                    raise ValueError("请求路径不能包含查询、片段或外部地址")
                path = unquote(url.path, errors="strict")
                parts = [p for p in path.split("/") if p]
                if path == "/":
                    self._send(200, page, "text/html; charset=utf-8")
                elif path == "/api/data":
                    self._send(200, api_payload, "application/json; charset=utf-8")
                elif len(parts) == 3 and parts[0] == "render" and parts[2].endswith(".jpg"):
                    frame_id, selection = parts[1], parts[2][:-len(".jpg")]
                    if frame_id not in store.frame_ids:
                        raise FileNotFoundError("未知帧")
                    if selection not in {"all", "raw"} and selection not in store.tracks_by_id:
                        raise FileNotFoundError("未知轨线")
                    data = store.render(frame_id, selection)
                    self._send(200, data, store.render_content_type(frame_id, selection))
                else:
                    self.send_error(404, "未列入查看器路由")
            except FileNotFoundError:
                self.send_error(404, "文件未列入查看器白名单或不存在")
            except (ValueError, UnicodeError):
                self.send_error(400, "路径被拒绝或数据无效")
            except OSError:
                self.send_error(404, "渲染数据不可读取")

        def do_HEAD(self) -> None:
            self.do_GET()

        def _read_only(self) -> None:
            self.send_error(405, "只读查看器，仅支持 GET / HEAD")

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _read_only

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读物体轨线查看器（组代表链），不修改任何数据")
    parser.add_argument("--json", type=Path,
                        help="temporal groups chain evidence JSON (chain_basis=group_representatives)；省略时为仅边界模式")
    parser.add_argument("--run", required=True, type=Path, help="Verified run directory with */frame.jpg and masks.npz")
    parser.add_argument("--port", type=int, default=8768, help="本机 HTTP 端口（默认 8768）")
    parser.add_argument("--boundaries", type=Path, help="可选的 visible_boundaries_v1 单帧边界候选 JSON")
    args = parser.parse_args(argv)
    if args.json is None and args.boundaries is None:
        parser.error("需要 --json 或 --boundaries 至少其一")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    try:
        server = create_server(args.json, args.run, args.port, boundaries=args.boundaries)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"无法启动查看器：{exc}\n")
    print(f"只读物体轨线查看器：http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print("组是派生审阅统计，不是已核验物体。按 Ctrl+C 停止。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    main()
