"""Read-only, loopback-only viewer for field_boundary_run_v1 results."""
from __future__ import annotations

import argparse
import json
import math
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>农田边界 · 离线结果查看器</title>
<style nonce="__NONCE__">
:root{color-scheme:light;--ink:#20342d;--muted:#63746c;--line:#dbe4dd;--green:#256448}
*{box-sizing:border-box}body{margin:0;background:#f3f6f2;color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{padding:25px 32px;background:#fff;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px;align-items:center}
h1{font-size:24px;letter-spacing:.03em;margin:0}h2{font-size:16px;margin:0 0 12px}p{margin:6px 0}.subtle{color:var(--muted);font-size:13px}.badge{border:1px solid #bbd6c5;color:var(--green);border-radius:20px;padding:5px 14px;white-space:nowrap}
main{max-width:1600px;margin:auto;padding:22px 32px}.warning{padding:13px 18px;border:1px solid #e5c87e;background:#fff6dc;border-radius:9px;color:#6c5015;margin-bottom:18px}
.error{color:#94252b;background:#ffebec;border:1px solid #e6a8aa;border-radius:8px;padding:12px 16px;margin-bottom:14px;white-space:pre-wrap;overflow-wrap:anywhere}[hidden]{display:none!important}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:17px}button,select{font:inherit;border:1px solid #c9d5cc;border-radius:7px;background:#fff;color:var(--ink);padding:8px 13px}select{flex:1;min-width:180px;max-width:690px}button{cursor:pointer}button:hover:not(:disabled){border-color:var(--green)}button:disabled{opacity:.4;cursor:not-allowed}button[aria-pressed=true]{background:var(--green);border-color:var(--green);color:#fff}:focus-visible{outline:3px solid #6f9ad5;outline-offset:3px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:20px;align-items:start}.card{background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden}.card-body{padding:19px}.image-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)}.image-bar button{padding:5px 12px}.canvas{min-height:360px;display:grid;place-items:center;background:#18251f}.canvas img{width:100%;max-height:68vh;object-fit:contain;display:block}.image-message{color:#d5e0d9;padding:25px}.caption{padding:11px 16px;font-size:13px;color:var(--muted);overflow-wrap:anywhere}
.legend{display:flex;gap:9px 18px;flex-wrap:wrap;padding:16px;border-top:1px solid var(--line);font-size:12px}.legend span{display:flex;align-items:center;gap:6px}.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}.crop{background:rgb(74,190,95)}.ground{background:rgb(197,151,92)}.woody_vegetation{background:rgb(36,112,77)}.water{background:rgb(53,157,231)}.ditch{background:rgb(240,110,38)}.person{background:rgb(242,65,75)}.vehicle{background:rgb(253,196,50)}.tarp{background:rgb(139,126,238)}.pole{background:rgb(229,119,190)}.conflict{background:rgb(220,65,220)}
aside{display:grid;gap:15px}.stage{font:700 27px/1.3 ui-monospace,monospace;overflow-wrap:anywhere;margin-bottom:10px}.value{white-space:pre-wrap;overflow-wrap:anywhere}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:14px}.metric strong{display:block;font-size:23px}.wide{grid-column:1/-1}dl{margin:0}dt{color:var(--muted);font-size:12px;margin-top:10px}dd{margin:2px 0}.small{font-size:13px}details{margin-top:17px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,monospace;max-height:360px;overflow:auto}.empty{padding:65px 25px;text-align:center}.empty h2{font-size:23px}footer{padding:20px 0;font-size:12px;color:var(--muted)}
.proposal-review{margin-top:17px}.proposal-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}.proposal-item{border:1px solid var(--line);border-radius:8px;padding:14px}.proposal-item h3{font-size:15px;margin:0 0 8px;overflow-wrap:anywhere}.proposal-state{font-weight:700}.instance-list{padding-left:20px;margin-bottom:0}.instance-list li+li{margin-top:10px}
@media(max-width:950px){.layout{grid-template-columns:1fr}aside{grid-template-columns:1fr 1fr}.canvas{min-height:220px}}
@media(max-width:620px){header{padding:18px;align-items:flex-start}h1{font-size:19px}main{padding:16px}aside{grid-template-columns:1fr}.badge{font-size:11px;padding:4px 8px}.toolbar select{order:-1;flex-basis:100%;max-width:none}}
</style>
</head>
<body>
<header><div><h1>农田边界 · 离线结果查看器</h1><p class="subtle">逐帧检查语义证据，不执行感知、不打开源视频、不改写结果或标注。</p></div><span class="badge">127.0.0.1 · 只读</span></header>
<main>
<div class="warning"><strong>候选不等于安全掉头区；当前无定位/深度。</strong> 图像边界仅是候选，不提供掉头授权。</div>
<div id="error" class="error" role="alert" hidden></div>
<div id="empty" class="card empty" hidden><h2>暂无可查看的帧</h2><p>manifest.json 中的 frames 为空。请先输出离线结果，再重新打开页面。</p><p class="subtle">查看器不会生成、补齐或伪造推理结果。</p></div>
<div id="viewer" hidden>
<div class="toolbar"><button id="previous" type="button">← 上一帧</button><label for="frames" class="subtle">选择帧</label><select id="frames" aria-label="选择帧"></select><button id="next" type="button">下一帧 →</button><span id="position" class="subtle" aria-live="polite"></span></div>
<div class="layout">
<section class="card" aria-label="帧图像">
<div class="image-bar"><strong>图像证据</strong><div><button id="original" type="button" aria-pressed="false">原图</button> <button id="overlay" type="button" aria-pressed="true">语义叠加</button> <button id="rejected-toggle" type="button" aria-pressed="false" hidden>显示被拒实例</button></div></div>
<div id="proposal-view-control" class="image-bar" hidden><label for="proposal-views">候选视图</label><select id="proposal-views" aria-label="候选叠加视图"></select></div>
<div class="canvas"><img id="image" alt="当前帧图像" hidden><span id="image-message" class="image-message">正在读取图像…</span></div>
<div id="caption" class="caption"></div>
<div id="proposal-legend" class="caption" hidden><strong>候选叠加图：</strong>青框 = 输入候选范围；白轮廓 = 空间一致；红轮廓 = 空间拒绝。显示整个 mask，完整保留、未按候选框裁剪（未 clip）。颜色只表示候选语义，不代表语义已核实。</div>
<div class="legend" aria-label="语义颜色图例">
<span><i class="swatch crop"></i>作物 crop</span><span><i class="swatch ground"></i>地面 ground</span><span><i class="swatch woody_vegetation"></i>木本植被</span><span><i class="swatch water"></i>水域</span><span><i class="swatch ditch"></i>沟渠</span><span><i class="swatch person"></i>人员</span><span><i class="swatch vehicle"></i>车辆</span><span><i class="swatch tarp"></i>篷布</span><span><i class="swatch pole"></i>杆体</span><span><i class="swatch conflict"></i>冲突</span>
</div></section>
<aside>
<section class="card card-body"><h2>阶段预测</h2><div id="stage" class="stage">UNKNOWN</div><div id="stage-reason" class="value small"></div><div id="reference" hidden><hr><strong class="small">人工参考阶段 · 仅对照</strong><div id="reference-stage" class="value"></div><div id="reviewed" class="value subtle"></div><div class="subtle">reviewed 按原字段显示，不代表查看器重新确认；标注不参与推理。</div></div></section>
<section class="card card-body"><h2>地头候选解释</h2><dl><dt>假设 / 状态</dt><dd id="hypothesis" class="value"></dd><dt>证据与限制</dt><dd id="reasons" class="value small"></dd><dt>机动验证 / 授权（原字段）</dt><dd id="authorization" class="value small"></dd><dt>几何模式 / 缺失信息</dt><dd id="geometry" class="value small"></dd></dl></section>
<section class="card card-body"><h2>结果概况</h2><div class="metrics"><div class="metric"><span id="unknown-label" class="subtle">未知比例</span><strong id="unknown">—</strong></div><div class="metric"><span class="subtle">冲突比例</span><strong id="conflict">—</strong></div><div class="metric wide"><span class="subtle">推理总时长</span><strong id="timing">—</strong></div></div><div id="unknown-explanation" class="warning value small" hidden></div><div id="model" class="value subtle"></div></section>
</aside>
</div>
<section id="proposal-review" class="card card-body proposal-review" hidden><h2>候选核验</h2><div class="warning"><strong id="proposal-source" class="value"></strong><p><strong id="proposal-source-warning">候选来自助手读图或人工审阅，不是本地 VLM 自动识别。</strong></p><p>空间一致不代表语义正确；车辆的自车候选关系未经核实，不删除或抑制障碍。实例分数不是语义或自车身份的核实结论。</p></div><div id="proposal-list" class="proposal-list"></div></section>
<details class="card card-body"><summary>语义区域 / 风险要素 / 边界证据（图像候选）</summary><pre id="evidence-details"></pre></details>
</div>
<footer>使用左右方向键切换帧；到首尾即停止，不循环、不自动播放。只展示磁盘中已有结果。更新结果后请重启查看器。</footer>
</main>
<script nonce="__NONCE__">
'use strict';
const $ = id => document.getElementById(id);
let frames = [], current = 0, mode = 'overlay', generation = 0, activeRequest;
let showRejected = false, currentEvidence = null;
const errors = new Map();
function report(key, message) {
  if (message) errors.set(key, message); else errors.delete(key);
  $('error').textContent = Array.from(errors.values()).join('\n');
  $('error').hidden = errors.size === 0;
}
const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
function text(value, fallback = '未提供') {
  if (value === undefined || value === null) return fallback;
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}
const put = (id, value, fallback) => { $(id).textContent = text(value, fallback); };
const percent = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? (value * 100).toFixed(1) + '%' : '未提供 / 无效';
async function getJSON(url, signal) {
  const response = await fetch(url, {cache: 'no-store', signal});
  if (!response.ok) throw new Error('HTTP ' + response.status + ' · ' + url);
  return response.json();
}
function fileURL(path) {
  if (typeof path !== 'string' || !path) return null;
  return '/files/' + path.split('/').map(encodeURIComponent).join('/');
}
function renderProposalViews(frame) {
  const views = Array.isArray(frame.proposal_views) ? frame.proposal_views : [];
  $('proposal-views').replaceChildren();
  const all = document.createElement('option');
  all.value = ''; all.textContent = '全部候选';
  $('proposal-views').appendChild(all);
  views.forEach((view, index) => {
    const option = document.createElement('option');
    option.value = String(index); option.textContent = text(view.label, view.proposal_id);
    $('proposal-views').appendChild(option);
  });
  $('proposal-views').value = '';
  $('proposal-view-control').hidden = views.length === 0;
  $('proposal-legend').hidden = views.length === 0;
}
function showImage() {
  const frame = frames[current], selected = $('proposal-views').value;
  const view = selected === '' ? null : frame.proposal_views?.[Number(selected)];
  const base = view || frame;
  const overlayPath = showRejected ? base.overlay_path : (base.overlay_passed_path || base.overlay_path);
  $('original').setAttribute('aria-pressed', String(mode === 'original'));
  $('overlay').setAttribute('aria-pressed', String(mode === 'overlay'));
  $('overlay').disabled = !overlayPath;
  updateRejectedToggle();
  $('image').hidden = true;
  $('image-message').hidden = false;
  const url = fileURL(mode === 'overlay' ? overlayPath : frame.image_path);
  report('image', '');
  if (!url) {
    $('image').removeAttribute('src');
    put('image-message', '本帧未提供此图像，请切换原图。');
    return;
  }
  put('image-message', '正在读取图像…');
  $('image').alt = (mode === 'overlay' ? '语义叠加：' : '原图：') + text(frame.frame_id);
  $('image').src = url;
}
function rejectedCount(proposalId) {
  const instances = Array.isArray(currentEvidence && currentEvidence.instances) ? currentEvidence.instances : [];
  return instances.filter(raw => {
    const instance = object(raw);
    return instance.status === 'spatial_rejected' && instance.box_mask_disagreement !== true
      && (proposalId === null || instance.proposal_id === proposalId);
  }).length;
}
function disagreementCount(proposalId) {
  const instances = Array.isArray(currentEvidence && currentEvidence.instances) ? currentEvidence.instances : [];
  return instances.filter(raw => {
    const instance = object(raw);
    return instance.status === 'spatial_rejected' && instance.box_mask_disagreement === true
      && (proposalId === null || instance.proposal_id === proposalId);
  }).length;
}
function updateRejectedToggle() {
  const frame = frames[current], selected = $('proposal-views').value;
  const view = selected === '' ? null : frame.proposal_views?.[Number(selected)];
  const base = view || frame;
  const hidden = rejectedCount(view ? view.proposal_id : null);
  const dis = disagreementCount(view ? view.proposal_id : null);
  const button = $('rejected-toggle');
  const relevant = mode === 'overlay' && !!base.overlay_passed_path && (hidden > 0 || showRejected);
  button.hidden = !relevant;
  if (relevant) {
    button.textContent = showRejected ? '仅显示干净视图' : '显示其余被拒实例（' + hidden + '）';
    button.setAttribute('aria-pressed', String(showRejected));
  }
  const cleanView = mode === 'overlay' && !showRejected && !!base.overlay_passed_path;
  $('proposal-legend').textContent = cleanView
    ? '候选叠加图（干净视图）：白轮廓 = 空间一致'
      + (dis > 0 ? '；橙轮廓 = 高分框外（疑似框过小，需人工核对，共 ' + dis + ' 个）' : '')
      + (hidden > 0 ? '；另有 ' + hidden + ' 个低置信被拒实例已折叠，被拒掩码在数据中完整保留，未删除、未裁剪，点击"显示其余被拒实例"可展开' : '')
      + '。青框 = 输入候选范围。颜色只表示候选语义，不代表语义已核实。'
    : '候选叠加图：青框 = 输入候选范围；白轮廓 = 空间一致；橙轮廓 = 高分框外（疑似框过小）；红轮廓 = 其余空间拒绝。显示整个 mask，完整保留、未按候选框裁剪（未 clip）。颜色只表示候选语义，不代表语义已核实。';
}
$('image').addEventListener('load', () => { $('image').hidden = false; $('image-message').hidden = true; });
$('image').addEventListener('error', () => {
  $('image').hidden = true;
  $('image-message').hidden = false;
  put('image-message', '图像加载失败');
  report('image', '图像加载失败：请检查该帧图片是否存在且可读取。');
});
const proposalSources = new Map([
  ['assistant_visual_review', '助手读图候选（assistant_visual_review）'],
  ['human_review', '人工审阅候选（human_review）'],
  ['local_vlm', '本地 VLM 候选（local_vlm）'],
  ['cloud_vlm', '云端 VLM 候选（cloud_vlm）'],
]);
const candidateStatuses = new Map([
  ['no_detection', '未返回检测'], ['spatial_candidate', '空间候选（语义未核实）'],
  ['spatial_rejected', '空间拒绝'], ['spatially_consistent', '空间一致（语义未核实）'],
  ['outside_proposal', '空间拒绝（超出候选范围）'], ['empty_mask', '空 mask'],
]);
const statusText = value => (candidateStatuses.get(value) ?? '未识别状态') + ' · ' + text(value);
function appendText(parent, tag, value, className = 'value small') {
  const element = document.createElement(tag);
  element.className = className; element.textContent = text(value);
  parent.appendChild(element);
  return element;
}
function renderProposals(e) {
  const proposals = Array.isArray(e.proposals) ? e.proposals : [];
  const instances = Array.isArray(e.instances) ? e.instances : [];
  $('proposal-review').hidden = proposals.length === 0;
  $('proposal-list').replaceChildren();
  put('proposal-source', '来源：' + (proposalSources.get(e.proposal_source) ?? text(e.proposal_source)));
  $('proposal-source-warning').textContent = e.proposal_source === 'local_vlm'
    ? '候选由本地 VLM 自动生成，坐标与语义均未经人工确认，不代表核实结论或安全判断。'
    : e.proposal_source === 'cloud_vlm'
    ? '候选由云端 VLM 自动生成，坐标与语义均未经人工确认，不代表核实结论或安全判断。'
    : '候选来自助手读图或人工审阅，不是本地 VLM 自动识别。';
  if (proposals.length) $('proposal-legend').hidden = false;
  proposals.forEach(raw => {
    const p = object(raw), card = document.createElement('article');
    card.className = 'proposal-item';
    appendText(card, 'h3', text(p.proposal_id) + ' · ' + text(p.semantic));
    appendText(card, 'p', '状态：' + statusText(p.status), 'value proposal-state');
    const dis = instances.filter(raw => {
      const instance = object(raw);
      return instance.proposal_id === p.proposal_id && instance.box_mask_disagreement === true;
    }).length;
    appendText(card, 'p', '检测数：' + text(p.detection_count) + '；空间一致数：' + text(p.spatially_consistent_count)
      + (dis > 0 ? '；疑似框过小：' + dis : ''));
    appendText(card, 'p', '输入框 box_xyxy：' + text(p.box_xyxy) + '\n关系：' + text(p.relation)
      + (p.relation === 'ego_candidate' ? '（自车候选，未核实、不删除障碍）' : '')
      + '\nsemantic_verified: ' + text(p.semantic_verified));
    appendText(card, 'p', '候选理由：' + text(p.rationale) + '\n提示词：' + text(p.prompt));
    const matching = instances.filter(raw => object(raw).proposal_id === p.proposal_id);
    if (!matching.length) appendText(card, 'p', '未返回实例');
    else {
      const list = document.createElement('ul'); list.className = 'instance-list';
      matching.forEach((raw, index) => {
        const instance = object(raw), spatial = object(instance.spatial_check);
        const outside = spatial.outside_fraction === null && spatial.status === 'empty_mask'
          ? '不适用（空 mask）' : percent(spatial.outside_fraction);
        appendText(list, 'li', '实例 ' + (index + 1) + ' · proposal_id: ' + text(instance.proposal_id)
          + '\n分数：' + text(instance.score) + '\n空间检查：' + statusText(spatial.status)
          + '\n框外比例：' + outside + '\nsemantic_verified: ' + text(instance.semantic_verified)
          + '\nego_vehicle_verified: ' + text(instance.ego_vehicle_verified)
          + '\nsuppressed_as_ego: ' + text(instance.suppressed_as_ego)
          + (instance.box_mask_disagreement === true
            ? '\n疑似框过小：高分大区域被拒，分割可能正确，需人工核对' : ''));
      });
      card.appendChild(list);
    }
    $('proposal-list').appendChild(card);
  });
}
function renderEvidence(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('evidence 必须是 JSON 对象');
  currentEvidence = raw;
  const e = raw, h = object(e.hypothesis), g = object(e.geometry), m = object(e.model), t = object(e.timing);
  const candidateEvidence = e.region_proposals_used === true || proposalSources.has(e.proposal_source)
    || (Array.isArray(e.proposals) && e.proposals.length > 0);
  renderProposals(e);
  put('stage', e.stage_prediction, 'UNKNOWN');
  put('stage-reason', e.stage_reason);
  put('hypothesis', text(h.hypothesis) + ' / ' + text(h.status));
  put('reasons', Array.isArray(h.reasons) ? h.reasons.map(v => text(v)).join('\n') : h.reasons);
  put('authorization', 'maneuver_verified: ' + text(h.maneuver_verified) + '\nauthorization: ' + text(h.authorization));
  put('geometry', text(g.mode) + '\n缺失：' + text(g.missing));
  put('unknown-label', candidateEvidence ? '未知比例（语义未核实）' : '未知比例');
  put('unknown', percent(e.unknown_fraction));
  put('conflict', candidateEvidence && e.conflict_fraction === null ? '未计算（语义未核实）' : percent(e.conflict_fraction));
  $('unknown-explanation').hidden = !candidateEvidence;
  put('unknown-explanation', candidateEvidence
    ? '未知比例表示语义尚未核实，100% 未知不代表 100% 漏检，也不是漏检率。\nunknown_reason: ' + text(e.unknown_reason) : '');
  put('timing', typeof t.total_seconds === 'number' && Number.isFinite(t.total_seconds) && t.total_seconds >= 0 ? t.total_seconds.toFixed(3) + ' 秒' : '未提供');
  put('model', '后端：' + text(m.backend) + '\n分割后端使用 VLM：' + text(m.vlm_used) + '（候选来源以上方声明为准）\n提示版本：' + text(m.prompt_version));
  put('evidence-details', JSON.stringify(candidateEvidence ? e
    : {regions: e.regions ?? [], hazards: e.hazards ?? [], boundaries: e.boundaries ?? []}, null, 2));
  updateRejectedToggle();
}
function renderReference(frame) {
  const r = object(frame.annotation_reference);
  $('reference').hidden = !Object.hasOwn(frame, 'annotation_reference');
  put('reference-stage', r.stage, '未匹配到参考阶段');
  put('reviewed', 'reviewed（原字段）: ' + text(r.reviewed, '未提供') + '\n参考秒：' + text(r.second, '未匹配'));
}
async function selectFrame(index) {
  if (index < 0 || index >= frames.length) return;
  current = index;
  showRejected = false;
  const requestID = ++generation;
  if (activeRequest) activeRequest.abort();
  activeRequest = new AbortController();
  const frame = frames[current];
  $('frames').value = String(current);
  $('previous').disabled = current === 0; $('next').disabled = current === frames.length - 1;
  put('position', (current + 1) + ' / ' + frames.length);
  put('caption', text(frame.video_name) + ' · 帧 ' + text(frame.frame_index) + ' · ' + text(frame.capture_time_s) + ' 秒 · ' + text(frame.frame_id));
  renderProposalViews(frame);
  if (mode === 'overlay' && !frame.overlay_path) mode = 'original';
  renderEvidence({stage_reason: '正在读取证据…'}); renderReference(frame); showImage(); report('evidence', '');
  try {
    const url = fileURL(frame.evidence_path);
    if (!url) throw new Error('本帧未提供 evidence_path');
    const evidence = await getJSON(url, activeRequest.signal);
    if (requestID === generation) renderEvidence(evidence);
  } catch (error) {
    if (requestID === generation && error.name !== 'AbortError') {
      put('stage-reason', '证据读取失败；未展示上帧数据。');
      report('evidence', '证据加载失败：' + error.message);
    }
  }
}
$('frames').addEventListener('change', event => selectFrame(Number(event.target.value)));
$('previous').addEventListener('click', () => selectFrame(current - 1));
$('next').addEventListener('click', () => selectFrame(current + 1));
$('original').addEventListener('click', () => { mode = 'original'; showImage(); });
$('overlay').addEventListener('click', () => { mode = 'overlay'; showImage(); });
$('proposal-views').addEventListener('change', () => showImage());
$('rejected-toggle').addEventListener('click', () => { showRejected = !showRejected; showImage(); });
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || /^(SELECT|INPUT|TEXTAREA)$/.test(event.target.tagName) || event.target.isContentEditable) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault(); selectFrame(current + (event.key === 'ArrowRight' ? 1 : -1));
  }
});
(async () => {
  try {
    const manifest = await getJSON('/api/manifest');
    if (!Array.isArray(manifest.frames)) throw new Error('manifest.frames 不是数组');
    frames = manifest.frames;
    $('empty').hidden = frames.length !== 0;
    if (!frames.length) return;
    frames.forEach((frame, i) => {
      const option = document.createElement('option'); option.value = String(i);
      option.textContent = (i + 1) + '. ' + text(frame.video_name) + ' · ' + text(frame.capture_time_s) + ' 秒 · ' + text(frame.frame_id);
      $('frames').appendChild(option);
    });
    $('viewer').hidden = false;
    await selectFrame(0);
  } catch (error) { report('manifest', '清单加载失败：' + error.message); }
})();
</script>
</body></html>'''


def _relative_path(value: Any) -> str:
    """Reject non-canonical POSIX paths, including Windows drive/ADS syntax."""
    if not isinstance(value, str) or not value:
        raise ValueError("结果路径必须是非空的相对 POSIX 路径")
    if any(c in value for c in "\\:") or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("结果路径含非法字符")
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in value.split("/")):
        raise ValueError("结果路径不允许绝对路径、空段或路径遍历")
    return value


class ResultStore:
    """A startup snapshot of the manifest and its exact file allowlist."""

    def __init__(self, results: str | Path, annotations: str | Path | None = None):
        self.root = Path(results).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("--results 必须是目录")
        raw = json.loads(self.resolve("manifest.json").read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict) or raw.get("schema_version") != "field_boundary_run_v1":
            raise ValueError("manifest.schema_version 必须是 field_boundary_run_v1")
        if not isinstance(raw.get("frames"), list):
            raise ValueError("manifest.frames 必须是数组")
        self.allowed: dict[str, str] = {}
        frames = []
        # Project only viewer metadata: extra manifest fields cannot expose arbitrary files.
        for source in raw["frames"]:
            if not isinstance(source, dict):
                raise ValueError("manifest 中的每个帧必须是对象")
            frame = {key: source[key] for key in (
                "frame_id", "video_name", "frame_index", "capture_time_s",
                "image_path", "overlay_path", "overlay_passed_path", "evidence_path",
            ) if key in source}
            for key in ("image_path", "overlay_path", "overlay_passed_path", "evidence_path"):
                value = frame.get(key)
                if value is None or value == "":
                    continue
                rel = _relative_path(value)
                resolved = self.resolve(rel)  # Check containment even if the file does not yet exist.
                suffix = Path(rel).suffix.lower()
                if key == "evidence_path":
                    if suffix != ".json" or rel == "manifest.json":
                        raise ValueError("evidence_path 必须引用结果目录内的独立 JSON 文件")
                    mime = "application/json; charset=utf-8"
                else:
                    mime = self._image_type(rel, resolved)
                self.allowed[rel] = mime
            if "proposal_views" in source:
                views = source["proposal_views"]
                if not isinstance(views, list) or len(views) > 8:
                    raise ValueError("proposal_views 必须是最多 8 项的数组")
                frame["proposal_views"] = []
                for view in views:
                    if not isinstance(view, dict):
                        raise ValueError("proposal_views 每项必须是对象")
                    if (not isinstance(view.get("proposal_id"), str) or not view["proposal_id"]
                            or not isinstance(view.get("label"), str)):
                        raise ValueError("proposal_views 必须提供 proposal_id 和 label 文本")
                    rel = _relative_path(view.get("overlay_path"))
                    self.allowed[rel] = self._image_type(rel, self.resolve(rel))
                    record = {
                        "proposal_id": view["proposal_id"], "label": view["label"], "overlay_path": rel,
                    }
                    if view.get("overlay_passed_path") is not None and view["overlay_passed_path"] != "":
                        passed = _relative_path(view["overlay_passed_path"])
                        self.allowed[passed] = self._image_type(passed, self.resolve(passed))
                        record["overlay_passed_path"] = passed
                    frame["proposal_views"].append(record)
            frames.append(frame)
        self.manifest = {"schema_version": raw["schema_version"], "frames": frames}
        if "proposal_source" in raw:
            self.manifest["proposal_source"] = raw["proposal_source"]
        if annotations is not None:
            # Deliberately no default annotation location or directory search.
            annotation = json.loads(Path(annotations).read_text(encoding="utf-8-sig"))
            if not isinstance(annotation, dict):
                raise ValueError("标注文件必须是 JSON 对象")
            for frame in frames:
                frame["annotation_reference"] = self._reference(annotation, frame)

    def resolve(self, rel: str) -> Path:
        path = self.root.joinpath(*_relative_path(rel).split("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("结果路径或符号链接指向结果目录之外")
        return path

    def _image_type(self, rel: str, resolved: Path) -> str:
        # Check the target too: a .png symlink must not turn manifest/HTML into an image.
        mime = IMAGE_TYPES.get(Path(rel).suffix.lower())
        if (mime is None or resolved.suffix.lower() not in IMAGE_TYPES
                or resolved == self.resolve("manifest.json")):
            raise ValueError("image_path / overlay_path 只允许栅格图像，不能引用视频、HTML/SVG 或 manifest/JSON")
        return mime

    @staticmethod
    def _reference(annotation: dict, frame: dict) -> dict:
        reference = {}
        if "reviewed" in annotation:
            reference["reviewed"] = annotation["reviewed"]
        videos = annotation.get("videos")
        name = frame.get("video_name")
        video = videos.get(name) if isinstance(videos, dict) and isinstance(name, str) else None
        time_s = frame.get("capture_time_s")
        if not isinstance(video, dict) or isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
            return reference
        if not math.isfinite(time_s) or time_s < 0:
            return reference
        second = math.floor(time_s)
        stages = video.get("stages")
        if isinstance(stages, list) and second < len(stages):
            reference.update(stage=stages[second], second=second)
        if "reviewed" in video:
            reference["reviewed"] = video["reviewed"]
        return reference

    def read_file(self, rel: str) -> tuple[bytes, str]:
        # Re-resolve for every request: changing a symlink cannot expand the allowlist.
        _relative_path(rel)
        if rel not in self.allowed:
            raise FileNotFoundError("未列入清单")
        path = self.resolve(rel)
        if not path.is_file():
            raise FileNotFoundError("结果文件不存在")
        mime = self.allowed[rel]
        if mime.startswith("image/"):
            self._image_type(rel, path)
        data = path.read_bytes()
        if mime.startswith("application/json"):
            evidence = json.loads(data.decode("utf-8-sig"))
            if not isinstance(evidence, dict):
                raise ValueError("evidence 必须是 JSON 对象")
            data = json.dumps(evidence, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return data, mime


def create_server(results: str | Path, port: int = 8766,
                  annotations: str | Path | None = None) -> ThreadingHTTPServer:
    """Build a read-only server; port=0 is useful for isolated tests."""
    store = ResultStore(results, annotations)
    nonce = secrets.token_urlsafe(24)
    page = HTML.replace("__NONCE__", nonce).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "FieldBoundaryReview"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            # Never log attacker-controlled URL text or local filesystem paths.
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
            # BaseHTTPRequestHandler otherwise produces an HTML error page.
            self._send(code, json.dumps({"error": message or "请求未被允许"},
                                       ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:
            port = self.server.server_address[1]
            if self.headers.get("Host", "").lower() not in {
                f"127.0.0.1:{port}", f"localhost:{port}",
            }:
                self.send_error(403, "仅接受本机地址访问")
                return
            try:
                raw = self.path
                if not raw.startswith("/") or raw.startswith("//") or any(ord(c) < 32 or ord(c) == 127 for c in raw):
                    raise ValueError("非法请求路径")
                url = urlsplit(raw)
                if url.query or url.fragment or url.netloc or url.scheme:
                    raise ValueError("请求路径不能包含查询、片段或外部地址")
                path = unquote(url.path, errors="strict")
                if path == "/":
                    self._send(200, page, "text/html; charset=utf-8")
                elif path in ("/api/manifest", "/manifest.json"):
                    # Also check manifest containment after startup, though only the snapshot is served.
                    store.resolve("manifest.json")
                    self._send(200, json.dumps(store.manifest, ensure_ascii=False,
                                               allow_nan=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                elif path.startswith("/files/"):
                    data, mime = store.read_file(path[len("/files/"):])
                    self._send(200, data, mime)
                else:
                    self.send_error(404, "未列入结果白名单")
            except FileNotFoundError:
                self.send_error(404, "文件未列入结果白名单或不存在")
            except (ValueError, UnicodeError, RecursionError):
                self.send_error(400, "路径被拒绝或结果 JSON 无效")
            except OSError:
                self.send_error(404, "结果文件不可读取")

        def do_HEAD(self) -> None:
            self.do_GET()

        def _read_only(self) -> None:
            self.send_error(405, "只读查看器，仅支持 GET / HEAD")

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _read_only

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读本地农田边界结果查看器，不执行推理、不修改标注")
    parser.add_argument("--results", required=True, type=Path, help="包含 manifest.json 的离线结果目录")
    parser.add_argument("--port", type=int, default=8766, help="本机 HTTP 端口（默认 8766）")
    parser.add_argument("--annotations", type=Path, help="可选：显式指定人工阶段标注 JSON，仅只读对照")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    try:
        server = create_server(args.results, args.port, args.annotations)
    except (OSError, ValueError, RecursionError) as exc:
        parser.exit(2, f"无法启动查看器：{exc}\n")
    print(f"只读结果查看器：http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print("候选不等于安全掉头区；当前无定位/深度。按 Ctrl+C 停止。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
