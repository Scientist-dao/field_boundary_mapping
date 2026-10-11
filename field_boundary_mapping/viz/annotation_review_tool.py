# -*- coding: utf-8 -*-
"""地头视频逐秒标注可视化校对工具（纯标准库 + OpenCV 抽帧，无其他依赖）

用法:
    python annotation_review_tool.py [--port 8000]
然后浏览器打开 http://127.0.0.1:8000

说明:
    原视频编码为 MPEG-4 Part 2 (mp4v), 浏览器无法直接播放,
    因此界面用"每秒中间帧"图像代替视频, 与逐秒标签粒度一致。
    帧按需抽取并缓存在 视频抽帧_20260918/frame_cache/ 下。

功能:
    - 大图显示当前秒画面, ▶播放=1fps 步进, ←/→ ±1s
    - 时间条按秒着色显示阶段标签; 选阶段后拖动涂刷
    - 事件时刻一键"设为当前秒" / 不确定秒标记 / 逐段备注
    - "保存到文件"写回 地头数据集标注_v0_20260919.json (UTF-8, 缩进2)

仅监听 127.0.0.1, 视频从 ../地头数据集 只读抽帧, 不上传任何数据。
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
JSON_PATH = os.path.join(BASE, "地头数据集标注_v0_20260919.json")
VIDEO_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "地头数据集"))
CACHE_DIR = os.path.join(BASE, "视频抽帧_20260918", "frame_cache")

_extract_lock = threading.Lock()


def extract_frame(video_name, sec):
    """抽取 video_name 第 sec 秒的中间帧, 返回 JPEG bytes (带磁盘缓存)。"""
    stem = os.path.splitext(video_name)[0]
    cache = os.path.join(CACHE_DIR, stem, "s%02d.jpg" % sec)
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            return f.read()
    src = os.path.join(VIDEO_DIR, video_name)
    with _extract_lock:
        if os.path.isfile(cache):  # 双检: 等锁期间可能已被其他线程生成
            with open(cache, "rb") as f:
                return f.read()
        cap = cv2.VideoCapture(src)
        tmp = None
        if not cap.isOpened():
            # GBK 路径兜底: OpenCV 在 Windows 上偶发打不开非 ASCII 路径
            tmp = os.path.join(tempfile.gettempdir(), "anno_tool_copy.mp4")
            shutil.copyfile(src, tmp)
            cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        f_idx = min(int(round((sec + 0.5) * fps)), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(f_idx - 5, 0))
            ok, frame = cap.read()
        cap.release()
        if tmp:
            os.remove(tmp)
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return None
        data = buf.tobytes()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            f.write(data)
        return data


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>地头标注校对工具</title>
<style>
  :root { --bg:#1e1f22; --panel:#2b2d30; --line:#3c3f44; --fg:#dfe1e5; --dim:#9da0a8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:"Microsoft YaHei",system-ui,sans-serif; font-size:14px; }
  header { display:flex; align-items:center; gap:14px; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; }
  #saveBtn { background:#4caf50; color:#fff; border:0; padding:6px 18px;
             border-radius:4px; cursor:pointer; font-size:14px; }
  #saveBtn:disabled { background:#555; cursor:default; }
  #status { color:var(--dim); }
  .dirty #status::after { content:" ●未保存"; color:#ffb300; }
  main { display:flex; gap:16px; padding:14px 16px; flex-wrap:wrap; }
  #left { flex:1 1 480px; min-width:340px; }
  #right { flex:1 1 300px; min-width:280px; }
  #tabs { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }
  #tabs button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
                 padding:4px 10px; border-radius:4px; cursor:pointer; }
  #tabs button.on { background:#4c6ef5; border-color:#4c6ef5; }
  #frameWrap { background:#000; border-radius:6px; min-height:200px;
               display:flex; align-items:center; justify-content:center; }
  #frame { width:100%; max-height:62vh; display:block; border-radius:6px; }
  #ctrls { display:flex; gap:8px; align-items:center; margin:8px 0; flex-wrap:wrap; }
  #ctrls button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
                  border-radius:4px; padding:5px 14px; cursor:pointer; font-size:15px; }
  #ctrls button.on { background:#4c6ef5; border-color:#4c6ef5; }
  #meta { color:var(--dim); margin:4px 0 8px; }
  #curinfo { font-size:15px; margin:6px 0; }
  #tl { display:flex; flex-wrap:wrap; gap:3px; margin-bottom:8px; }
  .cell { width:36px; height:34px; border-radius:4px; cursor:pointer;
          display:flex; align-items:center; justify-content:center;
          font-size:12px; color:#111; user-select:none; position:relative;
          border:2px solid transparent; touch-action:none; }
  .cell.unc { border-color:#e53935; }
  .cell.cur { outline:2px solid #fff; }
  .cell.evt::after { content:"◆"; position:absolute; top:-2px; right:1px;
                     color:#fff; font-size:9px; text-shadow:0 0 2px #000; }
  #palette { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px; }
  #palette button { border:2px solid transparent; padding:5px 10px; border-radius:4px;
                    cursor:pointer; font-size:13px; color:#111; }
  #palette button.on { border-color:#fff; }
  #hint { color:var(--dim); margin:4px 0 10px; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:6px; padding:10px 12px; margin-bottom:12px; }
  .panel h3 { margin:0 0 8px; font-size:14px; }
  .evrow { display:flex; align-items:center; gap:8px; margin:5px 0; flex-wrap:wrap; }
  .evrow .nm { flex:0 0 110px; }
  .evrow .val { flex:0 0 44px; text-align:center; background:#111; border-radius:4px;
                padding:2px 0; }
  .evrow button { background:#333; color:var(--fg); border:1px solid var(--line);
                  border-radius:4px; padding:2px 8px; cursor:pointer; }
  textarea { width:100%; min-height:70px; background:#111; color:var(--fg);
             border:1px solid var(--line); border-radius:4px; padding:6px; }
  label.chk { color:var(--dim); }
  #uncbtn { margin-top:8px; }
</style>
</head>
<body>
<header>
  <h1>地头标注校对</h1>
  <button id="saveBtn">保存到文件</button>
  <span id="status"></span>
  <label class="chk"><input type="checkbox" id="reviewed"> 人工已校对(reviewed)</label>
</header>
<main>
  <div id="left">
    <div id="tabs"></div>
    <div id="frameWrap"><img id="frame" alt="当前秒画面"></div>
    <div id="ctrls">
      <button id="prevBtn">◀ 上一秒</button>
      <button id="playBtn">▶ 播放</button>
      <button id="nextBtn">下一秒 ▶</button>
      <span id="frameNo" style="color:var(--dim)"></span>
    </div>
    <div id="meta"></div>
    <div id="curinfo"></div>
    <div id="palette"></div>
    <div id="hint">选阶段后: 在时间条上按下并拖动 = 涂刷; 选"跳转"后: 点击秒格 = 跳到该秒。
      快捷键: ←/→ ±1s, 空格 播放/暂停, 1-4 把当前秒设为对应阶段(行内/接近行尾/地头/入新行)。</div>
    <div id="tl"></div>
  </div>
  <div id="right">
    <div class="panel"><h3>事件时刻</h3><div id="events"></div></div>
    <div class="panel"><h3>当前秒</h3>
      <div id="cursec"></div>
      <button id="uncbtn">标记/取消 不确定</button>
    </div>
    <div class="panel"><h3>本段备注</h3><textarea id="notes"></textarea></div>
  </div>
</main>
<script>
const STAGES=["IN_ROW","APPROACH","HEADLAND","ENTER_ROW"];
const STAGE_NAMES={IN_ROW:"行内",APPROACH:"接近行尾",HEADLAND:"地头",ENTER_ROW:"入新行"};
const COLORS={IN_ROW:"#66bb6a",APPROACH:"#ffee58",HEADLAND:"#a1887f",ENTER_ROW:"#42a5f5"};
const EVENTS=[["row_end_first_visible","行尾首次可见"],["exit_row_time","出行时刻"],
              ["headland_entry","进入地头"],["headland_exit","离开地头"],
              ["new_row_entry","入新行"]];
let ann=null, cur=null, curSec=0, paintStage=null, painting=false, dirty=false;
let playing=false, playTimer=null, frameReq=0;
const frame=document.getElementById('frame');

function setDirty(d){ dirty=d; document.body.classList.toggle('dirty',d);
  document.getElementById('status').textContent=d?'':'已保存 '+new Date().toLocaleTimeString(); }

function buildTabs(){
  const tabs=document.getElementById('tabs'); tabs.innerHTML='';
  Object.keys(ann.videos).forEach((name,i)=>{
    const b=document.createElement('button');
    b.textContent='V'+(i+1)+' '+ann.videos[name].capture_time.slice(5);
    b.title=name;
    b.onclick=()=>select(name);
    b.dataset.name=name; tabs.appendChild(b);
  });
}
function markTab(){
  document.querySelectorAll('#tabs button').forEach(b=>
    b.classList.toggle('on', b.dataset.name===cur));
}
function select(name){
  cur=name; const v=ann.videos[name];
  markTab();
  document.getElementById('meta').textContent=
    name+' · '+v.seconds_total+'s · 光照:'+v.environment.lighting+' · '+v.environment.note;
  document.getElementById('notes').value=v.notes||'';
  document.getElementById('reviewed').checked=!!ann.reviewed;
  curSec=0; stopPlay(); loadFrame(); renderAll();
}
function renderAll(){ renderTL(); renderEvents(); renderCur(); }

function loadFrame(){
  const req=++frameReq;
  document.getElementById('frameNo').textContent='第'+curSec+'秒 加载中…';
  frame.src='/frame/'+encodeURIComponent(cur)+'/'+curSec+'?r='+req;
}
frame.onload=()=>{ document.getElementById('frameNo').textContent=
  '第'+curSec+'秒 (该秒中间帧)'; };

function renderTL(){
  const tl=document.getElementById('tl'); tl.innerHTML='';
  const v=ann.videos[cur];
  const evtSecs=new Set(Object.values(v.events).filter(t=>t!==null));
  v.stages.forEach((st,i)=>{
    const c=document.createElement('div');
    c.className='cell'+(v.uncertain_seconds&&v.uncertain_seconds.includes(i)?' unc':'')
               +(evtSecs.has(i)?' evt':'')+(i===curSec?' cur':'');
    c.style.background=COLORS[st]; c.textContent=i;
    c.title='第'+i+'秒 · '+STAGE_NAMES[st];
    c.onpointerdown=(e)=>{ e.preventDefault();
      if(paintStage){ painting=true; applyPaint(i); } else seek(i); };
    c.onpointerenter=()=>{ if(painting&&paintStage) applyPaint(i); };
    tl.appendChild(c);
  });
}
document.addEventListener('pointerup',()=>{ painting=false; });

function applyPaint(i){
  const v=ann.videos[cur];
  if(v.stages[i]!==paintStage){ v.stages[i]=paintStage; setDirty(true); renderTL(); renderCur(); }
}
function seek(i){
  const n=ann.videos[cur].stages.length;
  i=Math.max(0,Math.min(n-1,i));
  if(i===curSec) return;
  curSec=i; loadFrame(); renderTL(); renderCur();
  if(playing && i>=n-1) stopPlay();
}

function renderEvents(){
  const box=document.getElementById('events'); box.innerHTML='';
  const ev=ann.videos[cur].events;
  EVENTS.forEach(([k,nm])=>{
    const row=document.createElement('div'); row.className='evrow';
    const val=document.createElement('span'); val.className='val';
    val.textContent=(ev[k]===null||ev[k]===undefined)?'无':ev[k]+'s';
    const b1=document.createElement('button'); b1.textContent='设为当前秒';
    b1.onclick=()=>{ ev[k]=curSec; setDirty(true); renderEvents(); renderTL(); };
    const b2=document.createElement('button'); b2.textContent='清除';
    b2.onclick=()=>{ ev[k]=null; setDirty(true); renderEvents(); renderTL(); };
    const nmEl=document.createElement('span'); nmEl.className='nm'; nmEl.textContent=nm;
    row.append(nmEl,val,b1,b2);
    box.appendChild(row);
  });
}
function renderCur(){
  const v=ann.videos[cur]; const st=v.stages[curSec];
  document.getElementById('curinfo').innerHTML=
    '第 <b>'+curSec+'</b> 秒 · <b style="color:'+COLORS[st]+'">'+STAGE_NAMES[st]+'</b>';
  document.getElementById('cursec').textContent=
    '第'+curSec+'秒: '+STAGE_NAMES[st]+
    (v.uncertain_seconds&&v.uncertain_seconds.includes(curSec)?' (不确定)':'');
}
function buildPalette(){
  const p=document.getElementById('palette');
  const mk=(label,st,color)=>{
    const b=document.createElement('button');
    b.textContent=label; if(color) b.style.background=color;
    b.onclick=()=>{ paintStage=(paintStage===st)?null:st;
      document.querySelectorAll('#palette button').forEach(x=>x.classList.remove('on'));
      if(paintStage===st) b.classList.add('on'); };
    p.appendChild(b);
  };
  mk('跳转模式',null,'#888');
  STAGES.forEach(st=>mk(STAGE_NAMES[st],st,COLORS[st]));
}

function stopPlay(){
  playing=false;
  if(playTimer){ clearInterval(playTimer); playTimer=null; }
  document.getElementById('playBtn').textContent='▶ 播放';
  document.getElementById('playBtn').classList.remove('on');
}
document.getElementById('playBtn').onclick=()=>{
  if(playing){ stopPlay(); return; }
  playing=true;
  document.getElementById('playBtn').textContent='⏸ 暂停';
  document.getElementById('playBtn').classList.add('on');
  playTimer=setInterval(()=>{
    const n=ann.videos[cur].stages.length;
    if(curSec>=n-1){ stopPlay(); return; }
    seek(curSec+1);
  },1000);
};
document.getElementById('prevBtn').onclick=()=>seek(curSec-1);
document.getElementById('nextBtn').onclick=()=>seek(curSec+1);

document.addEventListener('keydown',(e)=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  if(e.key==='ArrowLeft'){ e.preventDefault(); seek(curSec-1); }
  else if(e.key==='ArrowRight'){ e.preventDefault(); seek(curSec+1); }
  else if(e.key===' '){ e.preventDefault();
    document.getElementById('playBtn').click(); }
  else if(['1','2','3','4'].includes(e.key)){
    const st=STAGES[Number(e.key)-1]; ann.videos[cur].stages[curSec]=st;
    setDirty(true); renderTL(); renderCur(); }
});
document.getElementById('notes').addEventListener('input',(e)=>{
  ann.videos[cur].notes=e.target.value; setDirty(true); });
document.getElementById('reviewed').addEventListener('change',(e)=>{
  ann.reviewed=e.target.checked; setDirty(true); });
document.getElementById('uncbtn').onclick=()=>{
  const v=ann.videos[cur];
  v.uncertain_seconds=v.uncertain_seconds||[];
  const i=v.uncertain_seconds.indexOf(curSec);
  if(i>=0) v.uncertain_seconds.splice(i,1); else v.uncertain_seconds.push(curSec);
  v.uncertain_seconds.sort((a,b)=>a-b); setDirty(true); renderTL(); renderCur(); };
document.getElementById('saveBtn').onclick=()=>{
  const b=document.getElementById('saveBtn'); b.disabled=true; b.textContent='保存中…';
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
                     body:JSON.stringify(ann)})
    .then(r=>r.json()).then(j=>{
      b.disabled=false; b.textContent='保存到文件';
      if(j.ok){ setDirty(false); } else { alert('保存失败'); } })
    .catch(()=>{ b.disabled=false; b.textContent='保存到文件'; alert('保存失败'); });
};
window.addEventListener('beforeunload',(e)=>{ if(dirty){ e.preventDefault(); e.returnValue=''; } });

fetch('/api/annotations').then(r=>r.json()).then(a=>{ ann=a; buildTabs(); buildPalette();
  select(Object.keys(a.videos)[0]); setDirty(false); });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AnnoTool/1.1"

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if path == "/api/annotations":
            try:
                with open(JSON_PATH, "rb") as f:
                    self._send(200, f.read(), "application/json; charset=utf-8")
            except OSError:
                self._send(500, "annotation file missing", "text/plain")
            return
        if path.startswith("/frame/"):
            rest = urllib.parse.unquote(path[len("/frame/"):])
            name, _, sec = rest.rpartition("/")
            name = os.path.basename(name)
            if name not in (os.path.basename(p) for p in os.listdir(VIDEO_DIR)
                            if p.lower().endswith(".mp4")):
                self._send(404, "video not found", "text/plain")
                return
            try:
                sec = int(sec)
            except ValueError:
                self._send(400, "bad second", "text/plain")
                return
            if not (0 <= sec < 60 * 60):
                self._send(400, "second out of range", "text/plain")
                return
            data = extract_frame(name, sec)
            if data is None:
                self._send(500, "frame extraction failed", "text/plain")
                return
            self._send(200, data, "image/jpeg")
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/save":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, '{"ok":false,"error":"invalid json"}',
                       "application/json")
            return
        with open(JSON_PATH, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self._send(200, '{"ok":true}', "application/json")

    def log_message(self, fmt, *args):
        pass  # 静默访问日志, 只留错误


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    port = 8000
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    if not os.path.isfile(JSON_PATH):
        print("[ERROR] annotation file not found:", JSON_PATH)
        return 1
    if not os.path.isdir(VIDEO_DIR):
        print("[ERROR] video dir not found:", VIDEO_DIR)
        return 1
    os.makedirs(CACHE_DIR, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("annotation tool running -> http://127.0.0.1:%d" % port)
    print("videos from:", VIDEO_DIR)
    print("frame cache:", CACHE_DIR)
    print("saves to   :", JSON_PATH)
    print("press Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
