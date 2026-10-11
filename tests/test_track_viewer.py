"""Pure unittest coverage; every fixture below is synthetic, not SAM3 output."""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

import cv2
import numpy as np

VIEWER_PATH = (Path(__file__).resolve().parents[1] / "field_boundary_mapping"
               / "viz" / "serve_track_viewer.py")
sys.path.insert(0, str(VIEWER_PATH.parents[2]))
SPEC = importlib.util.spec_from_file_location("track_viewer_under_test", VIEWER_PATH)
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


def write_synthetic_fixture(base: Path) -> tuple[Path, Path]:
    """Build a two-frame synthetic run plus its group-representative chain JSON."""
    run_dir = base / "run"
    frames = [
        {"frame_id": "syn-frame-0", "second": 0,
         "instances": [
             {"instance_id": "i-veh-0", "mask_key": "m-veh", "semantic": "vehicle",
              "bbox_xyxy": [2, 2, 20, 20], "score": 0.9,
              "spatial_check": {"status": "spatially_consistent"}},
             {"instance_id": "i-tarp-0", "mask_key": "m-tarp", "semantic": "tarp",
              "bbox_xyxy": [5, 5, 12, 12], "score": 0.8,
              "spatial_check": {"status": "spatially_consistent"}},
         ]},
        {"frame_id": "syn-frame-1", "second": 2,
         "instances": [
             {"instance_id": "i-veh-1", "mask_key": "m-veh", "semantic": "vehicle",
              "bbox_xyxy": [3, 2, 21, 20], "score": 0.88,
              "spatial_check": {"status": "spatially_consistent"}},
         ]},
    ]
    masks = {
        "m-veh": np.zeros((24, 32), dtype=bool),
        "m-tarp": np.zeros((24, 32), dtype=bool),
    }
    masks["m-veh"][4:18, 4:18] = True
    masks["m-tarp"][6:12, 6:12] = True
    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    entries = []
    for frame in frames:
        frame_id = frame["frame_id"]
        frame_dir = run_dir / frame_id
        frame_dir.mkdir(parents=True)
        source = {"video_name": "synthetic.mp4",
                  "video_sha256": hashlib.sha256(b"synthetic video").hexdigest(),
                  "frame_index": frame["second"] * 30 + 15,
                  "capture_time_s": frame["second"] + 0.5}
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        (frame_dir / "frame.jpg").write_bytes(buffer.tobytes())
        (frame_dir / "evidence.json").write_text(
            json.dumps({"frame_id": frame_id, "source": source,
                        "image": {"width": 32, "height": 24},
                        "instances": frame["instances"]}, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(frame_dir / "masks.npz",
                            **{k: v for k, v in masks.items()})
        entry = {"frame_id": frame_id, **source,
                 "image_path": f"{frame_id}/frame.jpg", "evidence_path": f"{frame_id}/evidence.json",
                 "masks_path": f"{frame_id}/masks.npz"}
        entry["artifact_sha256"] = {
            entry[field]: hashlib.sha256((run_dir / entry[field]).read_bytes()).hexdigest()
            for field in ("image_path", "evidence_path", "masks_path")}
        entries.append(entry)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "completed", "frames": entries}), encoding="utf-8")
    vehicle_steps = [
        {"frame_id": "syn-frame-0", "second": 0, "semantic": "vehicle", "score": 0.9,
         "bbox_xyxy": [2.0, 2.0, 20.0, 20.0], "group_id": "g-000",
         "member_count": 2, "passed_count": 1, "instance_id": "i-veh-0"},
        {"frame_id": "syn-frame-1", "second": 2, "semantic": "vehicle", "score": 0.88,
         "bbox_xyxy": [3.0, 2.0, 21.0, 20.0], "group_id": "g-000",
         "member_count": 1, "passed_count": 1, "instance_id": "i-veh-1"},
    ]
    tarp_step = {"frame_id": "syn-frame-0", "second": 0, "semantic": "tarp", "score": 0.8,
                 "bbox_xyxy": [5.0, 5.0, 12.0, 12.0], "group_id": "g-001",
                 "member_count": 1, "passed_count": 1, "instance_id": "i-tarp-0"}
    payload = {
        "schema_version": "temporal_consistency_v2", "run": "synthetic_run",
        "chain_basis": "group_representatives", "iou_threshold": 0.2,
        "sampled_seconds": [0, 2], "frames_total": 2,
        "annotation_used_for_inference": False, "semantic_verified": False,
        "chains": [
            {"chain_id": "chain-00", "length": 2, "seconds": [0, 2],
             "semantic_sequence": ["vehicle", "vehicle"], "flip_count": 0,
             "modal_semantic": "vehicle", "modal_fraction": 1.0, "appearance_rate": 1.0,
             "steps": vehicle_steps},
            {"chain_id": "chain-01", "length": 1, "seconds": [0],
             "semantic_sequence": ["tarp"], "flip_count": 0,
             "modal_semantic": "tarp", "modal_fraction": 1.0, "appearance_rate": 0.5,
             "steps": [tarp_step]},
        ],
        "per_frame": [
            {"frame_id": "syn-frame-0", "second": 0, "units": []},
            {"frame_id": "syn-frame-1", "second": 2, "units": []},
        ],
    }
    evidence_path = base / "chains.json"
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return evidence_path, run_dir


def synthetic_boundaries(run_dir: Path) -> dict:
    """Loader-return fixture with real source hashes but hand-authored geometry."""
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    entries = json.loads(manifest_bytes)["frames"]
    frames = []
    for index, entry in enumerate(entries):
        frames.append({
            "frame_id": entry["frame_id"], "capture_time_s": entry["capture_time_s"],
            "image": {"width": 32, "height": 24},
            "source": {key: entry[key] for key in
                       ("artifact_sha256", "video_name", "video_sha256", "frame_index")},
            "boundaries": [{
                "boundary_id": "B001", "type": "crop_ground_interface",
                "side_a": {"semantic": "crop", "instance_ids": [f"crop-{index}"],
                           "source_semantics": ["crop"]},
                "side_b": {"semantic": "ground", "instance_ids": [f"ground-{index}"],
                           "source_semantics": ["ground", "harvested_ground"]},
                "points_px": [[3.5, 4], [14.5, 4]], "length_px": 11.0,
                "closed": False, "status": "image_candidate",
                "endpoint_reasons": [["junction"], ["interface_end", "unknown"]],
                "semantic_verified": False, "physical_field_boundary_verified": False,
                "maneuver_verified": False,
            }],
            "coverage_contours": [],
            "diagnostics": [
                {"reason": reason, "semantic": "crop",
                 "polylines_px": [[[-0.5, -0.5], [2.5, -0.5]]], "length_px": 3.0}
                for reason in ("unknown", "conflict", "occlusion", "fov")
            ],
            "stats": {},
        })
    return {
        "schema_version": "visible_boundaries_v1", "run": "synthetic_run",
        "parameters": {}, "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "semantic_verified": False, "physical_field_boundary_verified": False,
        "maneuver_verified": False, "annotation_used_for_inference": False,
        "frames": frames,
    }


def boundary_loader_module(payload: dict) -> ModuleType:
    from field_boundary_mapping.evidence.extract_visible_boundaries import contained_file, source_manifest

    module = ModuleType("extract_visible_boundaries")
    module.load_boundary_payload = mock.Mock(return_value=payload)
    module.contained_file = contained_file
    module.source_manifest = source_manifest
    return module


def write_extracted_boundary_fixture(base: Path, run_dir: Path, extension: str = ".png",
                                     shape: tuple[int, int] = (24, 32)) -> Path:
    """Extract real boundaries from alternate, independently hashed source artifacts."""
    from field_boundary_mapping.evidence.extract_visible_boundaries import extract_run

    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    height, width = shape
    for index, entry in enumerate(manifest["frames"]):
        folder = run_dir / entry["frame_id"]
        evidence = json.loads((folder / "evidence.json").read_bytes())
        evidence["image"] = {"width": width, "height": height}
        masks = {}
        # Same instance IDs as the chains, but different mask keys AND pixels.
        for number, instance in enumerate(evidence["instances"]):
            key = "authorized-" + instance["mask_key"]
            instance["mask_key"] = key
            mask = np.zeros(shape, dtype=bool)
            mask[2 + number * 8:7 + number * 8, width - 9:width - 3] = True
            masks[key] = mask
        left = np.zeros(shape, dtype=bool)
        left[:, :width // 2] = True
        for semantic, mask in (("crop", left), ("ground", ~left)):
            key = "surface-" + semantic
            evidence["instances"].append({
                "instance_id": key, "mask_key": key, "semantic": semantic,
                "spatial_check": {"status": "spatially_consistent"}})
            masks[key] = mask
        image = np.full((*shape, 3), (20 + index * 10, 60, 180), dtype=np.uint8)
        ok, encoded = cv2.imencode(extension, image)
        assert ok
        (folder / ("authorized" + extension)).write_bytes(encoded.tobytes())
        (folder / "authorized-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        np.savez_compressed(folder / "authorized-masks.npz", **masks)
        entry.update({"image_path": f"{entry['frame_id']}/authorized{extension}",
                      "evidence_path": f"{entry['frame_id']}/authorized-evidence.json",
                      "masks_path": f"{entry['frame_id']}/authorized-masks.npz"})
        entry["artifact_sha256"] = {
            entry[field]: hashlib.sha256((run_dir / entry[field]).read_bytes()).hexdigest()
            for field in ("image_path", "evidence_path", "masks_path")}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path = base / "extracted-boundaries.json"
    path.write_text(json.dumps(extract_run(run_dir)), encoding="utf-8")
    return path


_DEFAULT_EVIDENCE = object()


class TrackViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track-viewer-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.evidence, self.run_dir = write_synthetic_fixture(self.base)
        self.start_server()

    def start_server(self, boundaries: Path | None = None, json_path: object = _DEFAULT_EVIDENCE):
        evidence = self.evidence if json_path is _DEFAULT_EVIDENCE else json_path
        self.server = viewer.create_server(evidence, self.run_dir, port=0, boundaries=boundaries)
        self.addCleanup(self.server.server_close)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 5)
        self.addCleanup(self.server.shutdown)

    def request(self, path: str, method: str = "GET", host: str | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path, headers={"Host": host or f"127.0.0.1:{self.port}"})
            response = connection.getresponse()
            return response.status, response.read(), response.getheader("Content-Type")
        finally:
            connection.close()

    def test_html_page_and_api_data(self):
        status, body, mime = self.request("/")
        self.assertEqual(status, 200)
        self.assertTrue(mime.startswith("text/html"))
        self.assertIn("物体轨线查看器", body.decode("utf-8"))
        status, body, mime = self.request("/api/data")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["chain_basis"], "group_representatives")
        self.assertFalse(data["semantic_verified"])
        tags = {t["tag"]: t for t in data["tracks"]}
        self.assertEqual(tags["T01"]["length"], 2)
        self.assertEqual(tags["T01"]["members"], 3)
        self.assertTrue(tags["S01"]["singleton"])
        self.assertEqual(len(data["frames"]), 2)
        self.assertIsNone(data["visible_boundaries"])

    def test_render_routes(self):
        status, body, mime = self.request("/render/syn-frame-0/all.jpg")
        self.assertEqual(status, 200)
        self.assertTrue(mime.startswith("image/jpeg"))
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR))
        status, body, _ = self.request("/render/syn-frame-1/chain-00.jpg")
        self.assertEqual(status, 200)
        self.assertGreater(len(body), 500)
        status, _, _ = self.request("/render/syn-frame-1/chain-01.jpg")
        self.assertEqual(status, 200)
        status, _, _ = self.request("/render/syn-frame-0/chain-missing.jpg")
        self.assertEqual(status, 404)
        status, _, _ = self.request("/render/no-such-frame/all.jpg")
        self.assertEqual(status, 404)

    def test_render_cache_returns_identical_bytes(self):
        _, first, _ = self.request("/render/syn-frame-0/chain-00.jpg")
        _, second, _ = self.request("/render/syn-frame-0/chain-00.jpg")
        self.assertEqual(first, second)

    def test_rejects_query_strings_and_writes_and_foreign_hosts(self):
        status, _, _ = self.request("/render/syn-frame-0/all.jpg?x=1")
        self.assertEqual(status, 400)
        status, _, _ = self.request("/api/data", method="POST")
        self.assertEqual(status, 405)
        status, _, _ = self.request("/api/data", method="DELETE")
        self.assertEqual(status, 405)
        status, _, _ = self.request("/api/data", host="evil.example:80")
        self.assertEqual(status, 403)

    def test_non_group_payload_is_rejected_at_startup(self):
        payload = json.loads(self.evidence.read_text(encoding="utf-8"))
        payload["chain_basis"] = "instances"
        other = self.base / "instances.json"
        other.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            viewer.create_server(other, self.run_dir, port=0)

    def test_optional_boundaries_api_preserves_original_fields(self):
        _, original, _ = self.request("/api/data")
        original = json.loads(original)
        payload = synthetic_boundaries(self.run_dir)
        payload["frames"][1]["boundaries"] = []
        module = boundary_loader_module(payload)
        path = self.base / "boundaries.json"
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}):
            self.start_server(path)
        module.load_boundary_payload.assert_called_once_with(path, self.run_dir)
        status, body, _ = self.request("/api/data")
        self.assertEqual(status, 200)
        actual = json.loads(body)
        self.assertEqual(actual.pop("visible_boundaries"), payload)
        original.pop("visible_boundaries")
        self.assertEqual(actual, original)
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        self.assertEqual(self.request("/render/syn-frame-0/chain-00.jpg")[0], 200)
        self.assertEqual(self.request("/render/syn-frame-0/raw.jpg")[1],
                         (self.run_dir / "syn-frame-0" / "frame.jpg").read_bytes())
        self.assertEqual(self.request("/api/data", method="PUT")[0], 405)
        self.assertEqual(self.request("/api/data", host="evil.example")[0], 403)
        self.assertEqual(self.request("/boundaries.json")[0], 404)

    def test_raw_routes_return_source_bytes_without_loading_masks(self):
        with mock.patch.object(viewer.TrackStore, "_layer_mask", side_effect=AssertionError("no masks")):
            for frame_id in ("syn-frame-0", "syn-frame-1"):
                with self.subTest(frame_id=frame_id):
                    status, body, mime = self.request(f"/render/{frame_id}/raw.jpg")
                    self.assertEqual(status, 200)
                    self.assertEqual(mime, "image/jpeg")
                    self.assertEqual(body, (self.run_dir / frame_id / "frame.jpg").read_bytes())
            status, body, _ = self.request("/render/syn-frame-0/raw.jpg", method="HEAD")
            self.assertEqual((status, body), (200, b""))
        for path, expected in (("/render/missing/raw.jpg", 404),
                               ("/render/%2e%2e/raw.jpg", 404),
                               ("/render/syn-frame-0/raw.jpg?x=1", 400)):
            self.assertEqual(self.request(path)[0], expected)
        self.assertEqual(self.request("/render/syn-frame-0/raw.jpg", method="POST")[0], 405)
        self.assertEqual(self.request("/render/syn-frame-0/raw.jpg", host="evil.example")[0], 403)

    def test_optional_mode_keeps_security_headers(self):
        module = boundary_loader_module(synthetic_boundaries(self.run_dir))
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}):
            self.start_server(self.base / "boundaries.json")
        policies = []
        for path in ("/", "/api/data", "/render/syn-frame-0/raw.jpg"):
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                policy = response.getheader("Content-Security-Policy")
                policies.append(policy)
                self.assertRegex(policy, r"^default-src 'none'; img-src 'self'; connect-src 'self'; "
                                 r"script-src 'nonce-[\w-]+'; style-src 'nonce-[\w-]+'; "
                                 r"base-uri 'none'; form-action 'none'; frame-ancestors 'none'$")
                self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
                self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
                self.assertEqual(response.getheader("Cache-Control"), "no-store")
                response.read()
            finally:
                connection.close()
        self.assertEqual(len(set(policies)), 1)

    def test_real_manifest_png_sources_and_verified_cache_drive_raw_and_tracks(self):
        # No loader/source mocks: extract and validate actual PNG/JSON/NPZ files.
        path = write_extracted_boundary_fixture(self.base, self.run_dir)
        payload = json.loads(path.read_bytes())
        self.assertTrue(all(frame["boundaries"] for frame in payload["frames"]))
        manifest = json.loads((self.run_dir / "manifest.json").read_bytes())
        store = viewer.TrackStore(self.evidence, self.run_dir, path)
        self.start_server(path)
        originals, decoded, rendered = {}, {}, {}
        for entry in manifest["frames"]:
            frame_id = entry["frame_id"]
            raw = (self.run_dir / entry["image_path"]).read_bytes()
            originals[frame_id] = raw
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            decoded[frame_id] = image
            legacy = viewer.TrackStore(self.evidence, self.run_dir)
            self.assertEqual(image.shape, legacy._read_frame(frame_id).shape)
            self.assertFalse(np.array_equal(image, legacy._read_frame(frame_id)))
            np.testing.assert_array_equal(store._read_frame(frame_id), image)
            self.assertEqual(store.render(frame_id, "raw"), raw)
            self.assertEqual(self.request(f"/render/{frame_id}/raw.jpg"), (200, raw, "image/png"))
            self.assertEqual(self.request(f"/render/{frame_id}/raw.jpg", method="HEAD"),
                             (200, b"", "image/png"))
            evidence = json.loads((self.run_dir / entry["evidence_path"]).read_bytes())
            with np.load(self.run_dir / entry["masks_path"], allow_pickle=False) as masks:
                for layer in store.observations[frame_id]:
                    step = layer["step"]
                    key = next(i["mask_key"] for i in evidence["instances"]
                               if i["instance_id"] == step["instance_id"])
                    np.testing.assert_array_equal(store._layer_mask(frame_id, step), masks[key])
                    self.assertFalse(np.array_equal(masks[key], legacy._layer_mask(frame_id, step)))
            for selection in ("all", "chain-00", "chain-01"):
                data = store.render(frame_id, selection)
                rendered[frame_id, selection] = data
                overlay = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                # This unmasked corner must retain the authorized image, not frame.jpg.
                np.testing.assert_allclose(overlay[-1, 0], image[-1, 0], atol=3)

        # Mutate every source before the HTTP server's first track rendering.
        for entry in manifest["frames"]:
            ok, changed = cv2.imencode(".png", np.full((24, 32, 3), 240, dtype=np.uint8))
            self.assertTrue(ok)
            (self.run_dir / entry["image_path"]).write_bytes(changed.tobytes())
            (self.run_dir / entry["evidence_path"]).write_bytes(b"{}")
            (self.run_dir / entry["masks_path"]).write_bytes(b"changed archive")
            (self.run_dir / entry["frame_id"] / "frame.jpg").unlink()
        (self.run_dir / "manifest.json").write_bytes(b"{}")
        store._render_cache.clear()  # Re-render from source cache, not old JPEG results.
        for frame_id, raw in originals.items():
            with self.subTest(frame_id=frame_id):
                np.testing.assert_array_equal(store._read_frame(frame_id), decoded[frame_id])
                self.assertEqual(store.render(frame_id, "raw"), raw)
                self.assertEqual(self.request(f"/render/{frame_id}/raw.jpg"), (200, raw, "image/png"))
                for selection in ("all", "chain-00", "chain-01"):
                    expected = rendered[frame_id, selection]
                    self.assertEqual(store.render(frame_id, selection), expected)
                    self.assertEqual(self.request(f"/render/{frame_id}/{selection}.jpg"),
                                     (200, expected, "image/jpeg"))

    def test_real_manifest_jpeg_raw_uses_authorized_bytes(self):
        path = write_extracted_boundary_fixture(self.base, self.run_dir, extension=".jpg")
        raw = (self.run_dir / "syn-frame-0" / "authorized.jpg").read_bytes()
        self.assertNotEqual(raw, (self.run_dir / "syn-frame-0" / "frame.jpg").read_bytes())
        self.start_server(path)
        self.assertEqual(self.request("/render/syn-frame-0/raw.jpg"), (200, raw, "image/jpeg"))
        self.assertEqual(self.request("/render/syn-frame-0/all.jpg")[2], "image/jpeg")

    def test_boundary_only_mode_without_chain_evidence(self):
        # Sub-second runs have no chain evidence; the viewer must serve boundaries alone.
        path = write_extracted_boundary_fixture(self.base, self.run_dir)
        payload = json.loads(path.read_bytes())
        store = viewer.TrackStore(None, self.run_dir, path)
        self.assertEqual(store.tracks, [])
        self.assertEqual(store.payload["chain_basis"], "boundary_only_no_temporal_chains")
        sorted_frames = sorted(payload["frames"], key=lambda f: f["capture_time_s"])
        expected = [{"frame_id": f["frame_id"], "second": f["capture_time_s"]} for f in sorted_frames]
        self.assertEqual(store.frames, expected)
        self.assertEqual(store.frame_ids, {f["frame_id"] for f in payload["frames"]})
        self.start_server(path, json_path=None)
        status, body, _ = self.request("/api/data")
        self.assertEqual(status, 200)
        api = json.loads(body)
        self.assertEqual(api["tracks"], [])
        self.assertEqual([f["frame_id"] for f in api["frames"]], [f["frame_id"] for f in expected])
        self.assertEqual([f["second"] for f in api["frames"]], [f["capture_time_s"] for f in sorted_frames])
        self.assertIsNotNone(api["visible_boundaries"])
        manifest = json.loads((self.run_dir / "manifest.json").read_bytes())
        entries = {e["frame_id"]: e for e in manifest["frames"]}
        first_id = expected[0]["frame_id"]
        status, raw, _ = self.request(f"/render/{first_id}/raw.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(raw, (self.run_dir / entries[first_id]["image_path"]).read_bytes())

    def test_boundary_only_mode_requires_boundaries_json(self):
        with self.assertRaises(ValueError):
            viewer.TrackStore(None, self.run_dir)

    def test_boundary_html_contract(self):
        # Static contract checks, not a browser/UI integration test.
        html = viewer.HTML
        for label in ("边界候选", "原始图像", "候选轨线", "显示未知/冲突/遮挡/视野截断",
                      "无两侧支持的交界候选，不代表无边界或可通行",
                      "冠层/沟渠轮廓≠物理田界，裸地≠安全掉头区",
                      "显示闭合候选 GC（假设几何，非观测接触）",
                      "绿点线 GC：窄 unknown 带按就近闭合的假设几何，不是观测交界，永不升级 B"):
            self.assertIn(label, html)
        self.assertIn('<input id="showdiagnostics" type="checkbox">', html)
        self.assertIn('<input id="showgapclosed" type="checkbox">', html)
        self.assertIn('hasBoundaries ? "boundaries" : "tracks"', html)
        self.assertIn('svg.setAttribute("viewBox", "-0.5 -0.5 "', html)
        self.assertIn('state.mode === "tracks" ? state.selection : "raw"', html)
        boundary_code = html.split("function renderBoundaries()", 1)[1].split("function renderFilmstrip()", 1)[0]
        self.assertNotIn("innerHTML", boundary_code)
        for field in ("instance_ids", "source_semantics", "endpoint_reasons", "capture_time_s",
                      "gap_closed_boundaries", "gapclosedline", "assigned_neighbor_px", "width_px"):
            self.assertIn(field, boundary_code)
        self.assertIn('boundary.status === "gap_closed_candidate"', boundary_code)
        caption_code = html.split("function refresh()", 1)[1].split("function semanticLabel", 1)[0]
        self.assertIn("闭合候选 GC", caption_code)
        navigation = html.split("function setFrame(index)", 1)[1].split("function renderSelectionValue", 1)[0]
        self.assertIn("state.boundary = null", navigation)

    def test_gap_closed_candidates_flow_to_api(self):
        payload = synthetic_boundaries(self.run_dir)
        payload["frames"][0]["gap_closed_boundaries"] = [{
            "boundary_id": "GC001", "type": "crop_ground_interface",
            "side_a": {"semantic": "crop", "instance_ids": ["crop-0"], "source_semantics": ["crop"]},
            "side_b": {"semantic": "ground", "instance_ids": ["ground-0"], "source_semantics": ["ground"]},
            "points_px": [[8.5, 4], [12.5, 4]], "length_px": 4.0, "closed": False,
            "status": "gap_closed_candidate",
            "endpoint_reasons": [["interface_end"], ["interface_end"]],
            "gap": {"assigned_neighbor_px": 8, "side_a_assigned_px": 4, "side_b_assigned_px": 4,
                    "width_px": {"min": 2.0, "mean": 2.5, "max": 3.0}},
            "semantic_verified": False, "physical_field_boundary_verified": False,
            "maneuver_verified": False,
        }]
        module = boundary_loader_module(payload)
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}):
            store = viewer.TrackStore(self.evidence, self.run_dir, self.base / "boundaries.json")
        frame = store.api_data()["visible_boundaries"]["frames"][0]
        self.assertEqual(frame["gap_closed_boundaries"][0]["boundary_id"], "GC001")
        self.assertEqual(frame["gap_closed_boundaries"][0]["status"], "gap_closed_candidate")
        self.assertEqual(frame["gap_closed_boundaries"][0]["gap"]["width_px"]["max"], 3.0)


class TrackStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track-store-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.evidence, self.run_dir = write_synthetic_fixture(self.base)

    def test_tags_follow_length_order_and_singletons_last(self):
        store = viewer.TrackStore(self.evidence, self.run_dir)
        tags = [t["tag"] for t in store.tracks]
        self.assertEqual(tags[0], "T01")
        self.assertEqual(tags[1], "S01")
        self.assertEqual(store.tracks[0]["chain_id"], "chain-00")
        self.assertEqual(store.frame_size, (32, 24))
        self.assertEqual(len(store.observations["syn-frame-0"]), 2)
        self.assertEqual(len(store.observations["syn-frame-1"]), 1)

    def test_render_selection_dims_other_tracks(self):
        store = viewer.TrackStore(self.evidence, self.run_dir)
        all_bytes = store.render("syn-frame-0", "all")
        selected_bytes = store.render("syn-frame-0", "chain-00")
        self.assertNotEqual(all_bytes, selected_bytes)
        ghost_only = store.render("syn-frame-1", "chain-01")
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(ghost_only, np.uint8), cv2.IMREAD_COLOR))

    def test_no_boundary_dependency_without_option(self):
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": None}):
            store = viewer.TrackStore(self.evidence, self.run_dir)
            self.assertIsNone(store.api_data()["visible_boundaries"])
            self.assertEqual(store.render("syn-frame-0", "raw"),
                             (self.run_dir / "syn-frame-0" / "frame.jpg").read_bytes())
            ok, changed = cv2.imencode(".jpg", np.full((24, 32, 3), 200, dtype=np.uint8))
            self.assertTrue(ok)
            (self.run_dir / "syn-frame-0" / "frame.jpg").write_bytes(changed.tobytes())
            self.assertEqual(store.render("syn-frame-0", "raw"), changed.tobytes())
            np.testing.assert_array_equal(store._read_frame("syn-frame-0"),
                                          np.full((24, 32, 3), 200, dtype=np.uint8))

    def test_real_manifest_sets_frame_size_without_legacy_images(self):
        path = write_extracted_boundary_fixture(self.base, self.run_dir, shape=(26, 34))
        for frame_id in ("syn-frame-0", "syn-frame-1"):
            (self.run_dir / frame_id / "frame.jpg").unlink()
        store = viewer.TrackStore(self.evidence, self.run_dir, path)
        self.assertEqual(store.frame_size, (34, 26))
        self.assertEqual(store.api_data()["frame_size"], {"width": 34, "height": 26})
        self.assertEqual(store._read_frame("syn-frame-1").shape, (26, 34, 3))
        self.assertTrue(store.render("syn-frame-0", "all"))

    def test_real_manifest_artifact_hash_mismatches_are_rejected(self):
        path = write_extracted_boundary_fixture(self.base, self.run_dir)
        entry = json.loads((self.run_dir / "manifest.json").read_bytes())["frames"][0]
        for field in ("image_path", "evidence_path", "masks_path"):
            artifact = self.run_dir / entry[field]
            original = artifact.read_bytes()
            with self.subTest(field=field):
                try:
                    artifact.write_bytes(original + b"tampered")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        viewer.TrackStore(self.evidence, self.run_dir, path)
                finally:
                    artifact.write_bytes(original)

    def test_artifact_bytes_are_rechecked_after_loader_validation(self):
        import field_boundary_mapping.evidence.extract_visible_boundaries as extractor
        path = write_extracted_boundary_fixture(self.base, self.run_dir)
        entry = json.loads((self.run_dir / "manifest.json").read_bytes())["frames"][0]
        load = extractor.load_boundary_payload
        for field in ("image_path", "evidence_path", "masks_path"):
            artifact = self.run_dir / entry[field]
            original = artifact.read_bytes()

            def load_then_mutate(boundaries_path, run_dir):
                payload = load(boundaries_path, run_dir)
                artifact.write_bytes(original + b"changed after validation")
                return payload

            with self.subTest(field=field):
                try:
                    with mock.patch.object(extractor, "load_boundary_payload", side_effect=load_then_mutate):
                        with self.assertRaisesRegex(ValueError, "hash mismatch"):
                            viewer.TrackStore(self.evidence, self.run_dir, path)
                finally:
                    artifact.write_bytes(original)

    def test_manifest_is_bound_to_loader_result_before_caching(self):
        import field_boundary_mapping.evidence.extract_visible_boundaries as extractor
        path = write_extracted_boundary_fixture(self.base, self.run_dir)
        load = extractor.load_boundary_payload
        manifest = self.run_dir / "manifest.json"

        def load_then_mutate(boundaries_path, run_dir):
            payload = load(boundaries_path, run_dir)
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            return payload

        with mock.patch.object(extractor, "load_boundary_payload", side_effect=load_then_mutate):
            with self.assertRaisesRegex(ValueError, "does not match source run"):
                viewer.TrackStore(self.evidence, self.run_dir, path)

    def test_boundary_frame_mismatch_rejected_before_server_bind(self):
        for case in ("missing", "extra", "substituted", "duplicate"):
            with self.subTest(case=case):
                payload = synthetic_boundaries(self.run_dir)
                if case == "missing":
                    payload["frames"].pop()
                elif case == "extra":
                    payload["frames"].append({"frame_id": "not-in-chains"})
                elif case == "substituted":
                    payload["frames"][1]["frame_id"] = "not-in-chains"
                else:
                    payload["frames"].append(payload["frames"][0])
                module = boundary_loader_module(payload)
                with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}), \
                        mock.patch.object(viewer, "ThreadingHTTPServer") as server:
                    with self.assertRaisesRegex(ValueError, "frame_id"):
                        viewer.create_server(self.evidence, self.run_dir, port=0,
                                             boundaries=self.base / "boundaries.json")
                    server.assert_not_called()

    def test_boundary_dimensions_checked_for_every_frame(self):
        for index, dimension, value in ((0, "width", 31), (0, "height", 25),
                                         (1, "width", 31), (1, "height", 23),
                                         (0, "width", 32.0), (1, "height", True)):
            with self.subTest(index=index, dimension=dimension, value=value):
                payload = synthetic_boundaries(self.run_dir)
                payload["frames"][index]["image"][dimension] = value
                module = boundary_loader_module(payload)
                with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}):
                    with self.assertRaisesRegex(ValueError, "dimensions mismatch"):
                        viewer.TrackStore(self.evidence, self.run_dir, self.base / "boundaries.json")

    def test_boundary_dimensions_match_each_source_not_just_first_frame(self):
        ok, image = cv2.imencode(".jpg", np.full((26, 34, 3), 120, dtype=np.uint8))
        self.assertTrue(ok)
        (self.run_dir / "syn-frame-1" / "frame.jpg").write_bytes(image.tobytes())
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        entry = manifest["frames"][1]
        entry["artifact_sha256"][entry["image_path"]] = hashlib.sha256(image.tobytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        payload = synthetic_boundaries(self.run_dir)
        payload["frames"][1]["image"] = {"width": 34, "height": 26}
        module = boundary_loader_module(payload)
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}):
            store = viewer.TrackStore(self.evidence, self.run_dir, self.base / "boundaries.json")
        self.assertEqual(store.api_data()["visible_boundaries"], payload)

    def test_boundary_loader_validation_failure_propagates(self):
        module = boundary_loader_module(synthetic_boundaries(self.run_dir))
        module.load_boundary_payload.side_effect = ValueError("source hash mismatch")
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}), \
                mock.patch.object(viewer, "ThreadingHTTPServer") as server:
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                viewer.create_server(self.evidence, self.run_dir, port=0,
                                     boundaries=self.base / "boundaries.json")
            server.assert_not_called()

    def test_sampled_seconds_are_not_replaced_by_capture_time(self):
        chains = json.loads(self.evidence.read_text(encoding="utf-8"))
        chains["per_frame"][1]["second"] = 8
        chains["sampled_seconds"][1] = 8
        for chain in chains["chains"]:
            chain["seconds"] = [8 if second == 2 else second for second in chain["seconds"]]
            for step in chain["steps"]:
                if step["frame_id"] == "syn-frame-1":
                    step["second"] = 8
        self.evidence.write_text(json.dumps(chains), encoding="utf-8")
        payload = synthetic_boundaries(self.run_dir)
        payload["frames"][1]["capture_time_s"] = 8.5
        module = boundary_loader_module(payload)
        with mock.patch.dict(sys.modules, {"field_boundary_mapping.evidence.extract_visible_boundaries": module}):
            store = viewer.TrackStore(self.evidence, self.run_dir, self.base / "boundaries.json")
        data = store.api_data()
        self.assertEqual(data["sampled_seconds"], [0, 8])
        self.assertEqual(data["frames"][1]["second"], 8)
        self.assertEqual(data["visible_boundaries"]["frames"][1]["capture_time_s"], 8.5)
        caption_code = viewer.HTML.split("function frameTimeLabel(frame)", 1)[1].split("function setFrame", 1)[0]
        self.assertIn('"采样 " + frame.second', caption_code)
        self.assertIn('"s · 实际采集 " + boundary.capture_time_s', caption_code)

    def test_track_tag_anchor_is_deepest_pixel_inside_its_mask(self):
        concave = np.zeros((24, 32), dtype=bool)
        concave[9:23, 20:31] = True
        concave[9:14, 20:24] = False
        full = np.ones((24, 32), dtype=bool)
        tiny = np.zeros((24, 32), dtype=bool)
        tiny[23, 31] = True
        empty = np.zeros((24, 32), dtype=bool)
        for name, mask in (("concave", concave), ("full", full), ("tiny", tiny), ("empty", empty)):
            with self.subTest(mask=name):
                store = viewer.TrackStore(self.evidence, self.run_dir)
                before = store.api_data()
                with mock.patch.object(store, "_layer_mask", return_value=mask), \
                        mock.patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
                    store.render("syn-frame-1", "chain-00")
                self.assertEqual(store.api_data(), before)
                if not mask.any():
                    put_text.assert_not_called()
                    continue
                put_text.assert_called_once()
                self.assertEqual(put_text.call_args.args[1], "T01")
                x, y = put_text.call_args.args[2]
                self.assertTrue(mask[y, x])
                distance = cv2.distanceTransform(
                    np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
                self.assertEqual(distance[y, x], distance.max())
                self.assertNotEqual((x, y), (3, 2))  # Original bbox top-left.

    def test_cli_passes_optional_boundaries_path(self):
        for option in ([], ["--boundaries", str(self.base / "boundaries.json")]):
            with self.subTest(option=option):
                server = mock.Mock()
                server.server_address = ("127.0.0.1", 8768)
                server.serve_forever.side_effect = KeyboardInterrupt
                with mock.patch.object(viewer, "create_server", return_value=server) as create, \
                        mock.patch("builtins.print"):
                    result = viewer.main(["--json", str(self.evidence), "--run", str(self.run_dir)] + option)
                self.assertEqual(result, 0)
                create.assert_called_once_with(self.evidence, self.run_dir, 8768,
                                               boundaries=Path(option[1]) if option else None)
                server.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
