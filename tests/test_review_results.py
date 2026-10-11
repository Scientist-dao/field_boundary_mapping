"""Pure unittest coverage; every fixture below is synthetic, not SAM3 output."""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import quote
import zlib


VIEWER_PATH = Path(__file__).resolve().parents[1] / "review_results.py"
SPEC = importlib.util.spec_from_file_location("review_results_under_test", VIEWER_PATH)
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)

INJECTION = '</script><img src=x onerror="window.INJECTED=true"> 合成测试，非真实推理'


def write_synthetic_fixture(root: Path, count: int = 2) -> dict:
    """Write small, clearly named synthetic fixtures only inside a caller's temp dir."""
    (root / "images").mkdir(exist_ok=True)
    (root / "evidence").mkdir(exist_ok=True)

    def png(color: tuple[int, int, int]) -> bytes:
        def chunk(kind: bytes, content: bytes) -> bytes:
            return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff)
        width, height = 120, 80
        pixels = b"".join(b"\0" + b"".join(bytes(color if x < 70 else (197, 151, 92)) for x in range(width)) for _ in range(height))
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))

    frames = []
    for index in range(count):
        frame_id = f"synthetic-{index}"
        frame = dict(frame_id=frame_id, video_name="SYNTHETIC 合成测试（非真实推理）.mp4",
                     frame_index=index * 25, capture_time_s=float(index),
                     image_path=f"images/原图 {index}.png", overlay_path=f"images/overlay-{index}.png",
                     evidence_path=f"evidence/{frame_id}.json", future_key={"ignored": True})
        (root / frame["image_path"]).write_bytes(png((115, 142, 97)))
        (root / frame["overlay_path"]).write_bytes(png((74, 190, 95)))
        evidence = {
            "frame_id": frame_id, "image": {"width": 120, "height": 80},
            "stage_prediction": "UNKNOWN", "stage_reason": INJECTION if index == 0 else "合成测试：只有图像证据",
            "hypothesis": {"hypothesis": "headland_candidate", "status": "candidate",
                           "reasons": ["合成测试，不是真实模型输出", INJECTION],
                           "maneuver_verified": False, "authorization": "unknown"},
            "geometry": {"mode": "image_only", "missing": ["localization", "depth"]},
            "regions": [{"semantic": "crop", "score": 0.85, "area_fraction": 0.45,
                         "contours": [{"points": [[0, 0], [69, 0], [69, 79]], "hole": False}]}],
            "hazards": [], "boundaries": [{"type": "crop_edge", "polylines_px": [[[69, 0], [69, 79]]], "status": "image_candidate"}],
            "unknown_fraction": 0.125, "conflict_fraction": 0.025,
            "model": {"backend": "SYNTHETIC_TEST_ONLY", "vlm_used": False, "prompt_version": "test"},
            "timing": {"total_seconds": 0.123}, "future_extension": {"literal": INJECTION},
        }
        (root / frame["evidence_path"]).write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
        frames.append(frame)
    manifest = {"schema_version": "field_boundary_run_v1", "frames": frames, "future_schema_extension": True}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


class ReviewResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="field-review-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "results"
        self.root.mkdir()
        self.manifest = write_synthetic_fixture(self.root)
        self.servers = []
        self.addCleanup(self.stop_servers)

    def stop_servers(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def start(self, annotations=None):
        server = viewer.create_server(self.root, 0, annotations)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return server

    def request(self, server, path, method="GET", headers=None, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")

    def test_only_binds_loopback(self):
        self.assertEqual(self.start().server_address[0], "127.0.0.1")

    def test_manifest_and_whitelisted_files(self):
        server = self.start()
        for route in ("/api/manifest", "/manifest.json"):
            status, _, data = self.request(server, route)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["frames"][0]["frame_id"], "synthetic-0")
        for key in ("image_path", "overlay_path", "evidence_path"):
            rel = self.manifest["frames"][0][key]
            status, headers, data = self.request(server, "/files/" + quote(rel))
            self.assertEqual(status, 200)
            if key == "evidence_path":
                self.assertEqual(json.loads(data)["stage_prediction"], "UNKNOWN")
            else:
                self.assertEqual(headers["Content-Type"], "image/png")
                self.assertTrue(data.startswith(b"\x89PNG"))

    def test_unlisted_files_and_directories_are_not_served(self):
        (self.root / "private.json").write_text('{"secret":"not for viewer"}', encoding="utf-8")
        (self.root / "source.mp4").write_bytes(b"private video")
        server = self.start()
        for path in ("/files/private.json", "/files/source.mp4", "/files/", "/images/", "/evidence/",
                     "/source.mp4", "/files/manifest.json", "/annotation_review_tool.py", "/api/annotations"):
            with self.subTest(path=path):
                self.assertIn(self.request(server, path)[0], (400, 404))

    def test_encoded_traversal_and_windows_paths_are_rejected(self):
        server = self.start()
        paths = ["/files/../manifest.json", "/files/%2e%2e/secret", "/files/%2e%2e%2fsecret",
                 "/files/%252e%252e%252fsecret", "/files/..%5csecret", "/files/C:%5csecret",
                 "/files//absolute", "/files/evidence//synthetic-0.json", "/files/evidence/./synthetic-0.json",
                 "/files/images/overlay-0.png:stream", "/files/images/overlay-0.png%00",
                 "/files/%ff", "/files/%0aetc", "/files/evidence/synthetic-0.json?x=1",
                 "http://evil.example/files/evidence/synthetic-0.json"]
        for path in paths:
            with self.subTest(path=path):
                status, _, body = self.request(server, path)
                self.assertIn(status, (400, 403, 404))
                self.assertNotIn(str(self.root).encode(), body)

    def test_invalid_manifest_paths_fail_closed(self):
        for value in ("../outside.png", "/absolute.png", "C:/outside.png", "..\\outside.png",
                      "images//overlay-0.png", "images/./overlay-0.png", "images/overlay-0.png:stream", "images/../overlay.png"):
            with self.subTest(value=value):
                self.manifest["frames"][0]["image_path"] = value
                self.save_manifest()
                with self.assertRaises(ValueError):
                    viewer.ResultStore(self.root)

    def test_video_and_active_content_cannot_be_whitelisted_as_images(self):
        for value in ("source.mp4", "active.svg", "active.html", "script.js"):
            with self.subTest(value=value):
                self.manifest["frames"][0]["image_path"] = value
                self.save_manifest()
                with self.assertRaises(ValueError):
                    viewer.ResultStore(self.root)

    def make_link(self, link: Path, target: Path):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"OS does not permit symlink creation: {exc}")

    def test_symlink_outside_results_is_rejected(self):
        outside = self.base / "secret.png"
        outside.write_bytes(b"outside")
        self.make_link(self.root / "images" / "escape.png", outside)
        self.manifest["frames"][0]["image_path"] = "images/escape.png"
        self.save_manifest()
        with self.assertRaises(ValueError):
            viewer.ResultStore(self.root)

    def test_manifest_symlink_outside_results_is_rejected(self):
        outside = self.base / "outside.json"
        outside.write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.root / "manifest.json").unlink()
        self.make_link(self.root / "manifest.json", outside)
        with self.assertRaises(ValueError):
            viewer.ResultStore(self.root)

    def test_symlink_retarget_after_startup_is_rejected(self):
        original = self.root / self.manifest["frames"][0]["image_path"]
        link = self.root / "images" / "current.png"
        self.make_link(link, original)
        self.manifest["frames"][0]["image_path"] = "images/current.png"
        self.save_manifest()
        server = self.start()
        self.assertEqual(self.request(server, "/files/images/current.png")[0], 200)
        outside = self.base / "secret.png"
        outside.write_bytes(b"TOP_SECRET")
        link.unlink()
        self.make_link(link, outside)
        status, _, body = self.request(server, "/files/images/current.png")
        self.assertEqual(status, 400)
        self.assertNotIn(b"TOP_SECRET", body)

    @unittest.skipUnless(os.name == "nt", "Windows junction regression test")
    def test_windows_junction_cannot_escape_even_after_startup(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "overlay-0.png").write_bytes(b"TOP_SECRET")
        junction = self.root / "linked"

        def link(target):
            process = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                                     capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)

        link(self.root / "images")
        self.addCleanup(lambda: junction.rmdir() if junction.exists() else None)
        self.manifest["frames"][0]["overlay_path"] = "linked/overlay-0.png"
        self.save_manifest()
        server = self.start()
        self.assertEqual(self.request(server, "/files/linked/overlay-0.png")[0], 200)
        junction.rmdir()
        link(outside)
        status, _, body = self.request(server, "/files/linked/overlay-0.png")
        self.assertEqual(status, 400)
        self.assertNotIn(b"TOP_SECRET", body)
        with self.assertRaises(ValueError):
            viewer.ResultStore(self.root)

    def test_resolved_escape_check_without_symlink_privileges(self):
        store = viewer.ResultStore(self.root)
        with mock.patch.object(Path, "resolve", return_value=self.base / "outside.png"):
            with self.assertRaises(ValueError):
                store.read_file(self.manifest["frames"][0]["image_path"])

    def test_empty_manifest_is_supported(self):
        self.manifest["frames"] = []
        self.save_manifest()
        server = self.start()
        self.assertEqual(json.loads(self.request(server, "/api/manifest")[2])["frames"], [])
        page = self.request(server, "/")[2].decode()
        self.assertIn("暂无可查看的帧", page)
        self.assertIn("不会生成、补齐或伪造推理结果", page)

    def test_missing_and_invalid_manifest_are_actionable_startup_errors(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            viewer.ResultStore(self.root)
        for content in ("{}", "[]", "bad json", '{"schema_version":"field_boundary_run_v1","frames":{}}'):
            (self.root / "manifest.json").write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                viewer.ResultStore(self.root)

    def test_page_never_interpolates_untrusted_data(self):
        self.manifest["frames"][0]["frame_id"] = INJECTION
        self.save_manifest()
        server = self.start()
        status, headers, page = self.request(server, "/")
        html = page.decode()
        self.assertEqual(status, 200)
        self.assertNotIn(INJECTION, html)
        self.assertNotIn("innerHTML", html)
        self.assertNotIn("insertAdjacentHTML", html)
        self.assertIn("textContent", html)
        self.assertEqual(html.count("<script "), 1)
        self.assertIn("script-src 'nonce-", headers["Content-Security-Policy"])
        self.assertNotIn("unsafe-inline", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        manifest = json.loads(self.request(server, "/api/manifest")[2])
        self.assertEqual(manifest["frames"][0]["frame_id"], INJECTION)

    def test_extra_fields_are_tolerated_without_expanding_allowlist(self):
        self.manifest["frames"][0]["source_video_path"] = "source.mp4"
        self.save_manifest()
        server = self.start()
        data = json.loads(self.request(server, "/files/evidence/synthetic-0.json")[2])
        self.assertEqual(data["future_extension"]["literal"], INJECTION)
        projected = json.loads(self.request(server, "/api/manifest")[2])
        self.assertNotIn("source_video_path", projected["frames"][0])
        self.assertEqual(self.request(server, "/files/source.mp4")[0], 404)

    def test_missing_optional_keys_are_not_fabricated(self):
        self.manifest["frames"][0].pop("overlay_path")
        self.save_manifest()
        (self.root / "evidence/synthetic-0.json").write_text('{"frame_id":"synthetic-0","future":true}', encoding="utf-8")
        server = self.start()
        data = json.loads(self.request(server, "/files/evidence/synthetic-0.json")[2])
        self.assertNotIn("stage_prediction", data)
        self.assertNotIn("unknown_fraction", data)
        self.assertIn("e.stage_prediction, 'UNKNOWN'", viewer.HTML)
        self.assertIn("typeof value === 'number'", viewer.HTML)

    def test_missing_or_broken_evidence_returns_visible_fetch_error(self):
        server = self.start()
        path = self.root / "evidence/synthetic-0.json"
        for content in ("bad json", "[]", '{"unknown_fraction":NaN}'):
            path.write_text(content, encoding="utf-8")
            self.assertEqual(self.request(server, "/files/evidence/synthetic-0.json")[0], 400)
        path.unlink()
        self.assertEqual(self.request(server, "/files/evidence/synthetic-0.json")[0], 404)
        self.assertIn("if (!response.ok) throw new Error", viewer.HTML)
        self.assertIn("证据加载失败", viewer.HTML)

    def test_write_methods_rejected_and_files_unchanged(self):
        before = {p: hashlib.sha256(p.read_bytes()).digest() for p in self.root.rglob("*") if p.is_file()}
        server = self.start()
        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"):
            with self.subTest(method=method):
                self.assertEqual(self.request(server, "/api/manifest", method, body=b"{}")[0], 405)
        after = {p: hashlib.sha256(p.read_bytes()).digest() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_no_annotations_are_opened_by_default(self):
        original = Path.read_text
        opened = []

        def tracked(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", tracked):
            store = viewer.ResultStore(self.root)
        self.assertEqual(opened, [self.root / "manifest.json"])
        self.assertNotIn("annotation_reference", store.manifest["frames"][0])

    def test_explicit_annotations_preserve_reviewed_and_reference_only(self):
        path = self.base / "annotations.json"
        annotation = {"reviewed": False, "private": "must not be exposed", "videos": {
            self.manifest["frames"][0]["video_name"]: {"stages": ["APPROACH", "HEADLAND"]}}}
        path.write_text(json.dumps(annotation), encoding="utf-8")
        before = path.read_bytes()
        server = self.start(path)
        manifest = json.loads(self.request(server, "/api/manifest")[2])
        ref = manifest["frames"][1]["annotation_reference"]
        self.assertEqual(ref, {"reviewed": False, "stage": "HEADLAND", "second": 1})
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn("must not be exposed", json.dumps(manifest))
        self.assertEqual(self.request(server, "/files/annotations.json")[0], 404)
        evidence = json.loads(self.request(server, "/files/evidence/synthetic-0.json")[2])
        self.assertEqual(evidence["stage_prediction"], "UNKNOWN")

    def test_reference_uses_capture_second_without_clamping_or_guessing(self):
        annotation = {"videos": {"v.mp4": {"stages": ["IN_ROW", "HEADLAND"]}}}
        self.assertEqual(viewer.ResultStore._reference(annotation, {"video_name": "v.mp4", "capture_time_s": 1.9}),
                         {"stage": "HEADLAND", "second": 1})
        for time_s in (-1, 2, float("nan"), "1", True, None):
            with self.subTest(time_s=time_s):
                ref = viewer.ResultStore._reference(annotation, {"video_name": "v.mp4", "capture_time_s": time_s})
                self.assertNotIn("stage", ref)
                self.assertNotIn("reviewed", ref)

    def test_head_security_headers_and_rebinding_protection(self):
        server = self.start()
        status, headers, body = self.request(server, "/", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request(server, "/", headers={"Host": "evil.example"})[0], 403)

    def test_cli_port_validation(self):
        with mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit) as error:
            viewer.main(["--results", str(self.root), "--port", "0"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
