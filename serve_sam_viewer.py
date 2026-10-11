"""Read-only, loopback-only viewer for SAM3 scene refinement runs.

Serves the pre-rendered overlays (frame / coarse / refined / boundaries) for
every frame in a SAM refinement run directory. The viewer is evidence-only:
nothing is edited, no inference runs, and all verification flags stay false.
"""
from __future__ import annotations

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent

MODES = ("frame", "coarse", "refined", "boundaries")
MODE_FILENAMES = {
    "frame": "frame.jpg",
    "coarse": "coarse_overlay.jpg",
    "refined": "refined_overlay.jpg",
    "boundaries": "boundaries_overlay.jpg",
}
MODE_LABELS = {
    "frame": "原始帧",
    "coarse": "VLM 粗定位（框或多边形）",
    "refined": "SAM3 细化功能掩码",
    "boundaries": "边界叠加",
}


class RefinementStore:
    """Loads a SAM refinement run and serves its per-frame overlay images."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.frames: list[dict] = []
        self.frame_ids: list[str] = []
        self._frame_data: dict[str, dict[str, bytes]] = {}
        self._content_types: dict[str, str] = {}
        self._scene_evidence: dict[str, dict] = {}

        frame_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        for fd in frame_dirs:
            frame_id = fd.name
            images = {}
            for mode in MODES:
                path = fd / MODE_FILENAMES[mode]
                if path.exists():
                    images[mode] = path.read_bytes()
            if not images:
                continue
            scene_path = fd / "scene_evidence.json"
            evidence = {}
            if scene_path.exists():
                evidence = json.loads(scene_path.read_text(encoding="utf-8"))
            self._frame_data[frame_id] = images
            self._scene_evidence[frame_id] = evidence
            self._content_types[frame_id] = "image/jpeg"
            self.frames.append({"frame_id": frame_id})
            self.frame_ids.append(frame_id)

        if not self.frames:
            raise ValueError(f"No frame directories with overlay images found in {run_dir}")

    def api_data(self) -> dict:
        frames_info = []
        for f in self.frames:
            fid = f["frame_id"]
            ev = self._scene_evidence.get(fid, {})
            regions = ev.get("regions", [])
            frames_info.append({
                "frame_id": fid,
                "region_count": len(regions),
                "status": ev.get("status", "unknown"),
                "regions": [
                    {
                        "role": r.get("role", "unknown"),
                        "surface": r.get("surface", "unknown"),
                        "status": r.get("status", "unknown"),
                        "pixel_count": r.get("pixel_count"),
                        "coarse_support": r.get("coarse_support"),
                        "quality_flags": r.get("quality_flags", []),
                    }
                    for r in regions
                ],
            })
        return {
            "run": self.run_name,
            "frame_count": len(self.frames),
            "modes": list(MODES),
            "mode_labels": MODE_LABELS,
            "frames": frames_info,
            "semantic_verified": False,
            "physical_field_boundary_verified": False,
            "maneuver_verified": False,
            "note": "SAM3 细化可视化查看器 · 只读 · 不修改任何数据 · 所有核验标记恒为 false",
        }

    def frame_image(self, frame_id: str, mode: str) -> tuple[bytes, str]:
        if frame_id not in self._frame_data:
            raise FileNotFoundError(f"未知帧 {frame_id}")
        if mode not in self._frame_data[frame_id]:
            raise FileNotFoundError(f"帧 {frame_id} 无 {mode} 模式图像")
        return self._frame_data[frame_id][mode], self._content_types[frame_id]


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM3 细化查看器</title>
<style nonce="__NONCE__">
:root{color-scheme:light;--ink:#20342d;--muted:#63746c;--line:#dbe4dd;--green:#256448}
*{box-sizing:border-box}body{margin:0;background:#f3f6f2;color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{padding:22px 32px;background:#fff;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px;align-items:center}
h1{font-size:22px;margin:0}p{margin:4px 0 0}.subtle{color:var(--muted);font-size:13px}.badge{border:1px solid #bbd6c5;color:var(--green);border-radius:20px;padding:5px 14px;white-space:nowrap}
main{max-width:1680px;margin:auto;padding:20px 32px}
.warning{padding:12px 18px;border:1px solid #e5c87e;background:#fff6dc;border-radius:9px;color:#6c5015;margin-bottom:16px;font-size:14px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.toolbar .modelabel{margin-left:8px}
button,select{font:inherit;border:1px solid #c9d5cc;border-radius:7px;background:#fff;color:var(--ink);padding:7px 13px}button{cursor:pointer}button:hover{border-color:var(--green)}button[aria-pressed=true]{background:var(--green);border-color:var(--green);color:#fff}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:20px;align-items:start}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}.card-body{padding:16px}
.viewport{position:relative;background:#18251f}.viewport img{width:100%;display:block}
.caption{padding:10px 16px;font-size:13px;color:var(--muted);border-top:1px solid var(--line)}
.filmstrip{display:flex;gap:8px;overflow-x:auto;padding:12px 14px;border-top:1px solid var(--line);background:#fafcf9}
.thumb{flex:0 0 150px;border:2px solid transparent;border-radius:6px;overflow:hidden;padding:0;background:none;position:relative}
.thumb img{width:100%;display:block}
.thumb .sec{position:absolute;left:4px;top:4px;background:rgba(0,0,0,.75);color:#fff;font-size:11px;padding:1px 6px;border-radius:4px}
.thumb.active{border-color:var(--green)}
.regionlist{max-height:520px;overflow:auto}
.legend{display:flex;flex-wrap:wrap;gap:5px 12px;margin:0 0 10px;font-size:12px}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:-1px;border:1px solid rgba(0,0,0,.25)}
.blendhint{font-size:11px;color:var(--muted);margin:8px 0 10px;line-height:1.5}
.regionrow{display:flex;align-items:center;gap:8px;width:100%;padding:8px 10px;border:0;border-bottom:1px solid #eef3ef;background:#fff;text-align:left;font:inherit;cursor:default}
.regionrow:hover{background:#f4f8f5}
.rtag{font:700 12px/1 ui-monospace,monospace;border-radius:5px;padding:3px 7px;min-width:30px;text-align:center;color:#fff}
.rinfo{font:12px/1.3 ui-monospace,monospace;color:var(--muted);white-space:nowrap}
.rlabel{flex:1;font-size:13px}
.flag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;margin-left:4px;background:#fff3cd;color:#856404}
.flag.reject{background:#f8d7da;color:#721c24}
.flag.unrefined{background:#d1ecf1;color:#0c5460}
h2{font-size:16px;margin:0 0 10px}
footer{padding:18px 0 26px;font-size:12px;color:var(--muted)}
[hidden]{display:none!important}
@media(max-width:1000px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div><h1>SAM3 细化可视化查看器</h1><p class="subtle" id="subtitle">加载中…</p></div>
  <span class="badge">127.0.0.1 · 只读</span>
</header>
<main>
  <div class="warning"><strong>可视化仅用于人工审阅。</strong>所有核验标记恒为 false：semantic_verified、physical_field_boundary_verified、maneuver_verified 均不成立。不修改任何数据。</div>
  <div class="toolbar">
    <button id="prev" type="button">← 上一帧</button>
    <select id="frameselect" aria-label="选择帧"></select>
    <button id="next" type="button">下一帧 →</button>
    <span class="modelabel">模式：</span>
    <button id="mode_frame" type="button">原始帧</button>
    <button id="mode_coarse" type="button">VLM 粗粒度</button>
    <button id="mode_refined" type="button" aria-pressed="true">SAM3 细化</button>
    <button id="mode_boundaries" type="button">边界叠加</button>
    <span id="hint" class="subtle">←/→ 切帧 · 点击模式按钮切换叠加层</span>
  </div>
  <div class="layout">
    <section class="card">
      <div class="viewport"><img id="mainimage" alt="当前帧叠加图"></div>
      <div class="caption" id="maincaption"></div>
      <div class="filmstrip" id="filmstrip"></div>
    </section>
    <aside>
      <section class="card card-body"><h2>本帧功能区域</h2><div class="legend" id="rolelegend"></div><p class="blendhint">叠加图按上表角色色以 50% 半透明着色；多个区域重叠的像素显示各角色色的均值。图内颜色会随底图明暗偏移，请以左图轮廓对照右侧色板判断角色。</p><p class="subtle" id="regioncount"></p><div class="regionlist" id="regionlist"></div></section>
    </aside>
  </div>
  <footer>SAM3 细化查看器 · 只读 · 不执行推理 · 所有核验标记恒为 false</footer>
</main>
<script nonce="__NONCE__">
let DATA = null;
let state = { frame: 0, mode: "refined" };

const $ = (id) => document.getElementById(id);
const ROLE_LABELS = {current_field:"当前田块",neighbor_field:"邻田",headland_candidate:"地头候选",farm_road:"农路",obstacle:"障碍",unknown:"未知"};
const SURFACE_LABELS = {crop:"作物",harvested_ground:"收割地",ground:"地表",water:"水体",ditch:"沟渠",vehicle:"车辆",person:"人员",tarp:"篷布",woody_vegetation:"木本植被",pole:"杆",mixed:"混合",unknown:"未知"};
const STATUS_LABELS = {refined_candidate:"已细化",unrefined_surface:"未细化",spatial_rejected:"空间拒绝"};
const ROLE_COLORS = {current_field:"#0072B2",neighbor_field:"#009E73",headland_candidate:"#E69F00",farm_road:"#CC79A7",obstacle:"#D53700",unknown:"#AFAFAF"};

function setFrame(index) {
  state.frame = (index + DATA.frames.length) % DATA.frames.length;
  refresh();
}

function setMode(mode) {
  state.mode = mode;
  refresh();
}

function imgSrc(frame, mode) {
  return "/render/" + encodeURIComponent(frame.frame_id) + "/" + mode + ".jpg";
}

function refresh() {
  const frame = DATA.frames[state.frame];
  $("mainimage").src = imgSrc(frame, state.mode);
  $("mainimage").alt = DATA.mode_labels[state.mode] + " · " + frame.frame_id;
  $("frameselect").value = String(state.frame);
  for (const m of ["frame","coarse","refined","boundaries"]) {
    $("mode_" + m).setAttribute("aria-pressed", String(m === state.mode));
  }
  const caption = frame.frame_id + " · 帧 " + (state.frame + 1) + "/" + DATA.frames.length +
    " · " + DATA.mode_labels[state.mode] +
    " · " + frame.region_count + " 个区域" +
    " · " + frame.status;
  $("maincaption").textContent = caption;
  renderFilmstrip();
  renderRegions(frame);
}

function renderFilmstrip() {
  const strip = $("filmstrip");
  strip.innerHTML = "";
  for (let i = 0; i < DATA.frames.length; i++) {
    const frame = DATA.frames[i];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb" + (i === state.frame ? " active" : "");
    const img = document.createElement("img");
    img.src = imgSrc(frame, state.mode);
    img.alt = frame.frame_id;
    const sec = document.createElement("span");
    sec.className = "sec";
    sec.textContent = (i + 1) + "/" + DATA.frames.length;
    btn.appendChild(img);
    btn.appendChild(sec);
    btn.addEventListener("click", () => setFrame(i));
    strip.appendChild(btn);
  }
}

function renderRegions(frame) {
  const list = $("regionlist");
  list.innerHTML = "";
  $("regioncount").textContent = frame.region_count + " 个区域（" + frame.status + "）";
  if (!frame.regions.length) {
    const empty = document.createElement("p");
    empty.className = "subtle";
    empty.textContent = "本帧无功能区域";
    list.appendChild(empty);
    return;
  }
  frame.regions.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "regionrow";
    const tag = document.createElement("span");
    tag.className = "rtag";
    tag.style.background = ROLE_COLORS[r.role] || "#9E9E9E";
    tag.textContent = "r" + (i + 1);
    const label = document.createElement("span");
    label.className = "rlabel";
    label.textContent = (ROLE_LABELS[r.role] || r.role) + " · " + (SURFACE_LABELS[r.surface] || r.surface);
    const statusLabel = STATUS_LABELS[r.status] || r.status;
    if (r.status === "spatial_rejected") {
      const flag = document.createElement("span");
      flag.className = "flag reject";
      flag.textContent = statusLabel;
      label.appendChild(flag);
    } else if (r.status === "unrefined_surface") {
      const flag = document.createElement("span");
      flag.className = "flag unrefined";
      flag.textContent = statusLabel;
      label.appendChild(flag);
    }
    const info = document.createElement("span");
    info.className = "rinfo";
    let infoText = "";
    if (r.pixel_count != null) {
      const k = (r.pixel_count / 1000).toFixed(0) + "K";
      infoText += k;
    }
    if (r.coarse_support != null) {
      infoText += " · 支持 " + Math.round(r.coarse_support * 100) + "%";
    }
    info.textContent = infoText;
    row.append(tag, label, info);
    if (r.quality_flags && r.quality_flags.length) {
      const flagsRow = document.createElement("div");
      flagsRow.style.padding = "2px 10px 8px 55px";
      flagsRow.style.fontSize = "11px";
      flagsRow.style.color = "var(--muted)";
      flagsRow.textContent = r.quality_flags.join(" · ");
      list.appendChild(row);
      list.appendChild(flagsRow);
    } else {
      list.appendChild(row);
    }
  });
}

async function boot() {
  const response = await fetch("/api/data");
  DATA = await response.json();
  document.title = "SAM3 细化查看器 · " + DATA.run;
  const legend = $("rolelegend");
  for (const [role, color] of Object.entries(ROLE_COLORS)) {
    const item = document.createElement("span");
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = color;
    item.appendChild(sw);
    item.appendChild(document.createTextNode(ROLE_LABELS[role] || role));
    legend.appendChild(item);
  }
  $("subtitle").textContent = "run=" + DATA.run + " · " + DATA.frame_count + " 帧 · " + DATA.note;
  const select = $("frameselect");
  DATA.frames.forEach((frame, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = frame.frame_id + "（" + frame.region_count + " 区域）";
    select.appendChild(option);
  });
  select.addEventListener("change", () => setFrame(Number(select.value)));
  $("prev").addEventListener("click", () => setFrame(state.frame - 1));
  $("next").addEventListener("click", () => setFrame(state.frame + 1));
  for (const m of ["frame","coarse","refined","boundaries"]) {
    $("mode_" + m).addEventListener("click", () => setMode(m));
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      setFrame(state.frame + (event.key === "ArrowLeft" ? -1 : 1));
    }
  });
  refresh();
}

boot().catch((error) => {
  $("subtitle").textContent = "加载失败：" + error.message;
});
</script>
</body>
</html>
'''


def create_server(run_dir: Path, port: int = 8768) -> ThreadingHTTPServer:
    store = RefinementStore(run_dir)
    nonce = secrets.token_urlsafe(24)
    page = HTML.replace("__NONCE__", nonce).encode("utf-8")
    api_payload = json.dumps(store.api_data(), ensure_ascii=False, allow_nan=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "SAMRefineViewer"
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
                    frame_id, mode = parts[1], parts[2][:-len(".jpg")]
                    if frame_id not in store.frame_ids:
                        raise FileNotFoundError("未知帧")
                    if mode not in MODES:
                        raise FileNotFoundError("未知模式")
                    data, mime = store.frame_image(frame_id, mode)
                    self._send(200, data, mime)
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
    parser = argparse.ArgumentParser(description="SAM3 细化可视化只读查看器")
    parser.add_argument("--run", required=True, type=Path, help="SAM 细化 run 目录（含 */frame.jpg 等 overlay）")
    parser.add_argument("--port", type=int, default=8770, help="本机 HTTP 端口（默认 8770）")
    args = parser.parse_args(argv)
    if not args.run.is_dir():
        parser.error(f"--run 目录不存在：{args.run}")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    try:
        server = create_server(args.run, args.port)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"无法启动查看器：{exc}\n")
    print(f"SAM3 细化查看器：http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print("只读 · 按 Ctrl+C 停止。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    main()
