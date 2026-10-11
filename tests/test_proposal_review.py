"""Synthetic, offline viewer tests: no HTTP server, browser, or model is started."""
from __future__ import annotations

import base64
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


VIEWER_PATH = (Path(__file__).resolve().parents[1] / "field_boundary_mapping"
               / "evidence" / "review_results.py")
SPEC = importlib.util.spec_from_file_location("proposal_review_under_test", VIEWER_PATH)
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)

INJECTION = '</script><img src=x onerror="window.INJECTED=true"> 合成候选'
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII="
)


def synthetic_evidence(source="assistant_visual_review"):
    proposals = []
    for proposal_id, semantic, status, count, consistent in (
        ("crop-1", "crop", "spatial_candidate", 2, 1),
        ("vehicle-1", "vehicle", "spatial_rejected", 1, 0),
        ("ground-1", "ground", "no_detection", 0, 0),
    ):
        proposals.append({
            "proposal_id": proposal_id, "semantic": semantic, "box_xyxy": [0, 0, 10, 10],
            "relation": "ego_candidate" if semantic == "vehicle" else "unspecified",
            "rationale": INJECTION, "prompt": semantic, "status": status,
            "detection_count": count, "spatially_consistent_count": consistent,
            "semantic_verified": False,
        })
    instances = [
        {"proposal_id": proposal_id, "score": score,
         "status": "spatial_candidate_semantics_unverified" if status == "spatially_consistent" else "spatial_rejected",
         "box_mask_disagreement": disagreement,
         "spatial_check": {"status": status, "outside_fraction": outside},
         "semantic_verified": False, "ego_vehicle_verified": False, "suppressed_as_ego": False}
        for proposal_id, score, status, outside, disagreement in (
            ("crop-1", 0.85, "spatially_consistent", 0.05, False),
            ("crop-1", 0.4, "empty_mask", None, False),
            ("vehicle-1", 0.9, "outside_proposal", 0.7, True),
        )
    ]
    return {
        "schema_version": "field_boundary_frame_v1", "proposal_source": source,
        "proposals": proposals, "instances": instances, "region_proposals_used": True,
        "unknown_fraction": 1.0, "conflict_fraction": None,
        "unknown_reason": "All region semantics remain hypotheses; " + INJECTION,
        "stage_prediction": "UNKNOWN", "regions": [], "hazards": [], "boundaries": [],
        "model": {"backend": "SYNTHETIC_TEST_ONLY", "vlm_used": False},
        "future_extension": {"literal": INJECTION},
    }


class ReviewResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="proposal-review-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "results"
        self.root.mkdir()
        self.evidence = synthetic_evidence()
        self.frame = {
            "frame_id": "synthetic-0", "image_path": "original.png", "overlay_path": "all.png",
            "evidence_path": "evidence.json", "proposal_views": [
                {"proposal_id": p["proposal_id"], "label": INJECTION if i == 0 else p["semantic"],
                 "overlay_path": f"候选 {i}.png"}
                for i, p in enumerate(self.evidence["proposals"])
            ],
        }
        self.manifest = {"schema_version": "field_boundary_run_v1", "frames": [self.frame],
                         "proposal_source": self.evidence["proposal_source"]}
        for relative in ("original.png", "all.png", *(v["overlay_path"] for v in self.frame["proposal_views"])):
            (self.root / relative).write_bytes(PNG)
        self.save_manifest()
        self.save_evidence()

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")

    def save_evidence(self):
        (self.root / "evidence.json").write_text(json.dumps(self.evidence, ensure_ascii=False), encoding="utf-8")

    def test_proposal_views_whitelist_and_metadata_projection(self):
        self.frame["proposal_views"][0].update(image_path="private.png", evidence_path="private.json")
        self.frame["masks_path"] = "masks.npz"
        self.manifest["source_video_path"] = "source.mp4"
        self.save_manifest()
        for relative in ("private.png", "private.json", "masks.npz", "source.mp4", "unlisted.png"):
            (self.root / relative).write_bytes(b"not public")
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        store = viewer.ResultStore(self.root)
        projected = store.manifest["frames"][0]
        self.assertEqual(projected["image_path"], "original.png")
        self.assertEqual(projected["overlay_path"], "all.png")
        self.assertNotIn("masks_path", projected)
        self.assertNotIn("source_video_path", store.manifest)
        for original, view in zip(self.frame["proposal_views"], projected["proposal_views"]):
            self.assertEqual(view, {key: original[key] for key in ("proposal_id", "label", "overlay_path")})
            self.assertEqual(store.read_file(view["overlay_path"]), (PNG, "image/png"))
        self.assertEqual(set(store.allowed), {"original.png", "all.png", "evidence.json", "候选 0.png", "候选 1.png", "候选 2.png"})
        for relative in ("private.png", "private.json", "masks.npz", "source.mp4", "unlisted.png", "manifest.json"):
            with self.subTest(relative=relative), self.assertRaises(FileNotFoundError):
                store.read_file(relative)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_all_raster_suffixes_and_eight_views_are_supported(self):
        self.frame["proposal_views"] = []
        for i, suffix in enumerate((*viewer.IMAGE_TYPES, ".PNG", ".JPG")):
            relative = f"candidate-{i}{suffix}"
            (self.root / relative).write_bytes(PNG)  # Synthetic bytes; no decoder is invoked.
            self.frame["proposal_views"].append({"proposal_id": str(i), "label": str(i), "overlay_path": relative})
        self.save_manifest()
        store = viewer.ResultStore(self.root)
        self.assertEqual(len(store.manifest["frames"][0]["proposal_views"]), 8)
        for view in self.frame["proposal_views"]:
            relative = view["overlay_path"]
            self.assertEqual(store.read_file(relative)[1], viewer.IMAGE_TYPES[Path(relative).suffix.lower()])

    def test_passed_variant_paths_are_whitelisted_and_projected(self):
        self.frame["overlay_passed_path"] = "all-passed.png"
        for index, view in enumerate(self.frame["proposal_views"]):
            view["overlay_passed_path"] = f"passed-{index}.png"
        for relative in ("all-passed.png", "passed-0.png", "passed-1.png", "passed-2.png"):
            (self.root / relative).write_bytes(PNG)
        self.save_manifest()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        store = viewer.ResultStore(self.root)
        projected = store.manifest["frames"][0]
        self.assertEqual(projected["overlay_passed_path"], "all-passed.png")
        self.assertEqual(store.read_file("all-passed.png"), (PNG, "image/png"))
        for original, view in zip(self.frame["proposal_views"], projected["proposal_views"]):
            self.assertEqual(view["overlay_passed_path"], original["overlay_passed_path"])
            self.assertEqual(store.read_file(view["overlay_passed_path"]), (PNG, "image/png"))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_passed_variant_paths_reject_unsafe_values(self):
        self.frame["overlay_passed_path"] = "all.png"
        for relative in ("../outside.png", "/absolute.png", "C:\\outside.png", "images//view.png",
                         "active.html", "active.svg", "video.mp4", "evidence.json", "manifest.json",
                         "notes.txt", "no-extension", 7, ["view.png"], {"k": "v.png"}):
            with self.subTest(relative=relative):
                self.frame["proposal_views"][0]["overlay_passed_path"] = relative
                self.save_manifest()
                with self.assertRaises(ValueError):
                    viewer.ResultStore(self.root)
        self.frame["overlay_passed_path"] = "all-passed.png"
        (self.root / "all-passed.png").write_bytes(PNG)
        self.frame["proposal_views"][0]["overlay_passed_path"] = None
        self.frame["proposal_views"][1]["overlay_passed_path"] = ""
        self.save_manifest()
        store = viewer.ResultStore(self.root)
        projected = store.manifest["frames"][0]["proposal_views"]
        self.assertNotIn("overlay_passed_path", projected[0])
        self.assertNotIn("overlay_passed_path", projected[1])
        self.assertEqual(set(store.allowed),
                         {"original.png", "all.png", "all-passed.png", "evidence.json",
                          "候选 0.png", "候选 1.png", "候选 2.png"})

    def test_candidate_sources_and_raw_evidence_are_returned_without_rewriting(self):
        for source in ("assistant_visual_review", "human_review", "local_vlm", "cloud_vlm"):
            with self.subTest(source=source):
                self.evidence["proposal_source"] = self.manifest["proposal_source"] = source
                self.save_manifest()
                self.save_evidence()
                store = viewer.ResultStore(self.root)
                self.assertEqual(store.manifest["proposal_source"], source)
                data, mime = store.read_file("evidence.json")
                self.assertEqual(mime, "application/json; charset=utf-8")
                self.assertEqual(json.loads(data), self.evidence)
                self.assertIsNone(json.loads(data)["instances"][1]["spatial_check"]["outside_fraction"])

    def test_absent_and_empty_views_preserve_legacy_results(self):
        self.manifest.pop("proposal_source")
        self.evidence = {"unknown_fraction": 0.125, "conflict_fraction": 0.025, "future": INJECTION}
        self.save_evidence()
        for present in (False, True):
            with self.subTest(present=present):
                self.frame.pop("proposal_views", None)
                if present:
                    self.frame["proposal_views"] = []
                self.save_manifest()
                store = viewer.ResultStore(self.root)
                self.assertEqual(store.manifest, self.manifest)
                self.assertEqual(set(store.allowed), {"original.png", "all.png", "evidence.json"})
                self.assertEqual(json.loads(store.read_file("evidence.json")[0]), self.evidence)

    def test_invalid_view_shapes_and_more_than_eight_are_rejected(self):
        valid = self.frame["proposal_views"][0]
        invalid = [None, {}, "not a list", [valid] * 9, [None], ["view"],
                   [{}], [{**valid, "proposal_id": None}], [{**valid, "proposal_id": ""}],
                   [{**valid, "label": []}], [{k: v for k, v in valid.items() if k != "label"}],
                   [{k: v for k, v in valid.items() if k != "overlay_path"}]]
        for views in invalid:
            with self.subTest(views=views):
                self.frame["proposal_views"] = views
                self.save_manifest()
                with self.assertRaises(ValueError):
                    viewer.ResultStore(self.root)

    def test_proposal_paths_reject_traversal_absolute_ads_and_noncanonical_paths(self):
        for relative in (None, "", 7, "../outside.png", "images/../../outside.png", "/absolute.png",
                         "C:/outside.png", "C:\\outside.png", "..\\outside.png", "//host/share.png",
                         "images//view.png", "images/./view.png", "images/../view.png",
                         "view.png:stream", "image /view.png", "images./view.png", "view.png.",
                         "view.png ", "bad\x00.png", "bad\n.png", "bad\x7f.png"):
            with self.subTest(relative=relative):
                self.frame["proposal_views"][0]["overlay_path"] = relative
                self.save_manifest()
                with self.assertRaises(ValueError):
                    viewer.ResultStore(self.root)

    def test_proposal_paths_reject_active_content_video_and_manifest(self):
        for relative in ("active.html", "active.HTM", "active.svg", "active.SVG", "active.svgz",
                         "video.mp4", "video.AVI", "video.mov", "video.webm", "script.js",
                         "evidence.json", "manifest.json", "masks.npz", "no-extension"):
            with self.subTest(relative=relative):
                self.frame["proposal_views"][0]["overlay_path"] = relative
                self.save_manifest()
                with self.assertRaises(ValueError):
                    viewer.ResultStore(self.root)

    def test_resolved_targets_are_checked_at_startup_and_every_read(self):
        relative = self.frame["proposal_views"][0]["overlay_path"]
        original_resolve = Path.resolve
        for target in (self.base / "outside.png", self.root / "manifest.json",
                       self.root / "active.svg", self.root / "active.html", self.root / "video.mp4"):
            with self.subTest(target=target):
                store = viewer.ResultStore(self.root)

                def redirected(path, *args, **kwargs):
                    return target if path == self.root / relative else original_resolve(path, *args, **kwargs)

                # Exercise resolve-based protection even where Windows disallows symlink creation.
                with mock.patch.object(Path, "resolve", new=redirected):
                    with self.assertRaises(ValueError):
                        viewer.ResultStore(self.root)
                    # Non-image target must fail before read_bytes, even if it exists.
                    with mock.patch.object(Path, "is_file", return_value=True):
                        with self.assertRaises(ValueError):
                            store.read_file(relative)

    def test_manifest_resolved_to_image_named_target_is_never_served_as_image(self):
        relative = self.frame["proposal_views"][0]["overlay_path"]
        store = viewer.ResultStore(self.root)
        target = self.root / relative
        target.write_text(json.dumps(self.manifest), encoding="utf-8")
        original_resolve = Path.resolve

        def redirected(path, *args, **kwargs):
            return target if path == self.root / "manifest.json" else original_resolve(path, *args, **kwargs)

        # Simulate an internal manifest symlink to a misleading raster-suffixed file.
        with mock.patch.object(Path, "resolve", new=redirected):
            with self.assertRaises(ValueError):
                viewer.ResultStore(self.root)
            with self.assertRaises(ValueError):
                store.read_file(relative)

    def test_actual_symlink_escape_and_retarget_are_rejected(self):
        link = self.root / "linked.png"
        outside = self.base / "outside.png"
        outside.write_bytes(b"outside")
        try:
            link.symlink_to(self.root / "original.png")
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        self.frame["proposal_views"][0]["overlay_path"] = link.name
        self.save_manifest()
        store = viewer.ResultStore(self.root)
        self.assertEqual(store.read_file(link.name), (PNG, "image/png"))
        link.unlink()
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            store.read_file(link.name)
        with self.assertRaises(ValueError):
            viewer.ResultStore(self.root)

    def test_missing_proposal_image_does_not_expand_allowlist(self):
        store = viewer.ResultStore(self.root)
        relative = self.frame["proposal_views"][0]["overlay_path"]
        (self.root / relative).unlink()
        with self.assertRaises(FileNotFoundError):
            store.read_file(relative)
        self.assertEqual(viewer.ResultStore(self.root).allowed, store.allowed)

    def test_static_page_uses_text_content_and_has_candidate_safety_notices(self):
        self.assertNotIn("innerHTML", viewer.HTML)
        self.assertNotIn("insertAdjacentHTML", viewer.HTML)
        self.assertNotIn(INJECTION, viewer.HTML)
        self.assertIn("textContent", viewer.HTML)
        self.assertEqual(viewer.HTML.count('<script nonce="__NONCE__">'), 1)
        for notice in ("不是本地 VLM", "空间一致不代表语义正确", "不删除或抑制障碍",
                       "青框 = 输入候选范围", "白轮廓 = 空间一致", "橙轮廓 = 高分框外",
                       "红轮廓 = 其余空间拒绝", "未 clip", "显示其余被拒实例", "仅显示干净视图",
                       "干净视图", "未删除、未裁剪", "疑似框过小", "高分大区域", "需人工核对",
                       "低置信被拒实例"):
            self.assertIn(notice, viewer.HTML)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the in-memory UI test (no browser)")
    def test_ui_selection_candidate_evidence_and_legacy_behavior_without_browser(self):
        script = re.search(r'<script nonce="__NONCE__">(.*?)</script>', viewer.HTML, re.S).group(1)
        frames = [copy.deepcopy(self.frame) for _ in range(4)]
        for index, frame in enumerate(frames):
            frame.update(frame_id=f"synthetic-{index}", image_path=f"original-{index}.png",
                         overlay_path=f"all-{index}.png", evidence_path=f"evidence-{index}.json")
        frames[2].pop("proposal_views")
        frames[3]["proposal_views"] = []
        frames[3].pop("overlay_path")
        frames.append(copy.deepcopy(self.frame))
        frames[4].update(frame_id="synthetic-4", image_path="original-4.png",
                         overlay_path="all-4.png", evidence_path="evidence-4.json",
                         overlay_passed_path="all-4-passed.png")
        for view in frames[4]["proposal_views"]:
            view["overlay_passed_path"] = f"passed-{view['proposal_id']}.png"
        legacy = {"unknown_fraction": 0.125, "conflict_fraction": 0.025}
        payload = {"html": viewer.HTML, "script": script, "frames": frames,
                   "evidence": [self.evidence, synthetic_evidence("human_review"), legacy, legacy,
                                synthetic_evidence("local_vlm")],
                   "injection": INJECTION}
        check = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.literal = ''; this.value = ''; this.hidden = false; this.events = {}; this.attrs = {}; }
  set textContent(value) { this.literal = String(value); this.children = []; }
  get textContent() { return this.literal + this.children.map(child => child.textContent).join(''); }
  set innerHTML(value) { throw new Error('Unsafe dynamic HTML'); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; this.literal = ''; }
  setAttribute(name, value) { this.attrs[name] = value; }
  removeAttribute(name) { delete this.attrs[name]; delete this[name]; }
  addEventListener(name, callback) { this.events[name] = callback; }
  fire(name) { return this.events[name]({target: this}); }
}
const elements = new Map();
for (const match of input.html.matchAll(/<(\w+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
  const element = new Element(match[1]); element.hidden = /\bhidden\b/.test(match[0]);
  elements.set(match[2], element);
}
const el = id => { assert.ok(elements.has(id), id); return elements.get(id); };
const document = {getElementById: el, createElement: tag => new Element(tag), addEventListener() {}};
const responses = new Map([['/api/manifest', {frames: input.frames}]]);
input.frames.forEach((frame, i) => responses.set('/files/' + frame.evidence_path, input.evidence[i]));
const context = vm.createContext({document, console, AbortController, fetch: async url => {
  assert.ok(responses.has(url), 'Unexpected fetch: ' + url);
  return {ok: true, json: async () => responses.get(url)};
}});
(async () => {
  vm.runInContext(input.script, context);
  await new Promise(setImmediate);
  assert.equal(el('error').textContent, '');
  assert.equal(el('image').src, '/files/all-0.png');
  assert.equal(el('proposal-views').value, '');
  assert.equal(el('proposal-views').children.length, 4);
  assert.equal(el('proposal-views').children[0].textContent, '全部候选');
  assert.equal(el('proposal-views').children[1].textContent, input.injection);
  assert.equal(el('proposal-view-control').hidden, false);
  assert.equal(el('proposal-review').hidden, false);
  assert.equal(el('proposal-list').children.length, 3);
  assert.match(el('proposal-source').textContent, /助手读图.*assistant_visual_review/);
  for (const text of ['spatial_candidate', 'spatial_rejected', 'no_detection', '0.85', '5.0%', '70.0%',
                      '不适用（空 mask）', 'semantic_verified: false', 'ego_vehicle_verified: false',
                      'suppressed_as_ego: false', 'ego_candidate', '未返回实例', input.injection,
                      '疑似框过小：1', '疑似框过小：高分大区域被拒，分割可能正确，需人工核对']) {
    assert.ok(el('proposal-list').textContent.includes(text), text);
  }
  assert.equal(el('unknown').textContent, '100.0%');
  assert.match(el('unknown-label').textContent, /语义未核实/);
  assert.match(el('unknown-explanation').textContent, /不代表 100% 漏检/);
  assert.ok(el('unknown-explanation').textContent.includes(input.evidence[0].unknown_reason));
  assert.match(el('conflict').textContent, /未计算/);
  assert.deepEqual(JSON.parse(el('evidence-details').textContent), input.evidence[0]);
  const details = el('evidence-details').textContent;
  el('proposal-views').value = '0'; el('proposal-views').fire('change');
  assert.equal(el('image').src, '/files/' + encodeURIComponent(input.frames[0].proposal_views[0].overlay_path));
  el('original').fire('click');
  assert.equal(el('image').src, '/files/original-0.png');
  el('proposal-views').value = '1'; el('proposal-views').fire('change');
  assert.equal(el('image').src, '/files/original-0.png');
  assert.equal(el('original').attrs['aria-pressed'], 'true');
  el('overlay').fire('click');
  assert.equal(el('image').src, '/files/' + encodeURIComponent(input.frames[0].proposal_views[1].overlay_path));
  assert.equal(el('evidence-details').textContent, details);
  assert.equal(el('proposal-list').children.length, 3);
  await context.selectFrame(1);
  assert.equal(el('proposal-views').value, '');
  assert.equal(el('image').src, '/files/all-1.png');
  assert.equal(el('overlay').attrs['aria-pressed'], 'true');
  assert.match(el('proposal-source').textContent, /人工审阅.*human_review/);
  el('proposal-views').value = '1'; el('original').fire('click');
  await context.selectFrame(0);
  assert.equal(el('proposal-views').value, '');
  assert.equal(el('image').src, '/files/original-0.png');
  assert.equal(el('original').attrs['aria-pressed'], 'true');
  context.renderEvidence({...input.evidence[0], proposals: []});
  assert.equal(el('proposal-review').hidden, true);
  assert.equal(el('proposal-list').children.length, 0);
  el('overlay').fire('click');
  await context.selectFrame(2);
  assert.equal(el('image').src, '/files/all-2.png');
  for (const id of ['proposal-view-control', 'proposal-review', 'proposal-legend', 'unknown-explanation']) {
    assert.equal(el(id).hidden, true, id);
  }
  assert.equal(el('proposal-list').textContent, '');
  assert.equal(el('unknown-label').textContent, '未知比例');
  assert.equal(el('unknown').textContent, '12.5%');
  assert.equal(el('conflict').textContent, '2.5%');
  assert.deepEqual(JSON.parse(el('evidence-details').textContent), {regions: [], hazards: [], boundaries: []});
  await context.selectFrame(3);
  assert.equal(el('proposal-view-control').hidden, true);
  assert.equal(el('image').src, '/files/original-3.png');
  assert.equal(el('overlay').disabled, true);
  assert.equal(el('original').attrs['aria-pressed'], 'true');
  await context.selectFrame(4);
  assert.equal(el('proposal-view-control').hidden, false);
  assert.match(el('proposal-source').textContent, /本地 VLM 候选.*local_vlm/);
  assert.match(el('proposal-source-warning').textContent, /本地 VLM 自动生成/);
  assert.match(el('proposal-source-warning').textContent, /未经人工确认/);
  assert.equal(el('rejected-toggle').hidden, true);
  el('overlay').fire('click');
  assert.equal(el('image').src, '/files/all-4-passed.png');
  assert.equal(el('rejected-toggle').hidden, false);
  assert.equal(el('rejected-toggle').textContent, '显示其余被拒实例（1）');
  assert.equal(el('rejected-toggle').attrs['aria-pressed'], 'false');
  assert.match(el('proposal-legend').textContent, /干净视图/);
  assert.match(el('proposal-legend').textContent, /橙轮廓 = 高分框外/);
  assert.match(el('proposal-legend').textContent, /共 1 个/);
  assert.ok(el('proposal-legend').textContent.includes('未删除、未裁剪'));
  el('rejected-toggle').fire('click');
  assert.equal(el('image').src, '/files/all-4.png');
  assert.equal(el('rejected-toggle').textContent, '仅显示干净视图');
  assert.equal(el('rejected-toggle').attrs['aria-pressed'], 'true');
  assert.match(el('proposal-legend').textContent, /红轮廓 = 其余空间拒绝/);
  el('rejected-toggle').fire('click');
  assert.equal(el('image').src, '/files/all-4-passed.png');
  assert.equal(el('rejected-toggle').attrs['aria-pressed'], 'false');
  el('proposal-views').value = '1'; el('proposal-views').fire('change');
  assert.equal(el('image').src, '/files/' + encodeURIComponent(input.frames[4].proposal_views[1].overlay_passed_path));
  assert.equal(el('rejected-toggle').hidden, true);
  assert.match(el('proposal-legend').textContent, /橙轮廓 = 高分框外/);
  el('proposal-views').value = '0'; el('proposal-views').fire('change');
  assert.equal(el('image').src, '/files/' + encodeURIComponent(input.frames[4].proposal_views[0].overlay_passed_path));
  assert.equal(el('rejected-toggle').hidden, false);
  assert.equal(el('rejected-toggle').textContent, '显示其余被拒实例（1）');
  assert.ok(!el('proposal-legend').textContent.includes('橙轮廓'));
  el('rejected-toggle').fire('click');
  assert.equal(el('image').src, '/files/' + encodeURIComponent(input.frames[4].proposal_views[0].overlay_path));
  assert.equal(el('rejected-toggle').textContent, '仅显示干净视图');
  await context.selectFrame(1);
  assert.equal(el('rejected-toggle').hidden, true);
  assert.equal(el('image').src, '/files/all-1.png');
  await context.selectFrame(4);
  assert.equal(el('rejected-toggle').attrs['aria-pressed'], 'false');
  assert.equal(el('rejected-toggle').hidden, false);
  assert.equal(el('rejected-toggle').textContent, '显示其余被拒实例（1）');
  assert.equal(el('image').src, '/files/all-4-passed.png');
  assert.equal(el('error').textContent, '');
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
        completed = subprocess.run([shutil.which("node"), "-e", check], input=json.dumps(payload),
                                   text=True, encoding="utf-8", capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
