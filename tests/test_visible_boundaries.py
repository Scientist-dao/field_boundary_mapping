"""Synthetic, model-free regression tests for visible single-frame interfaces."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "field_boundary_mapping"
sys.path.insert(0, str(ROOT))
from field_boundary_mapping.evidence.extract_visible_boundaries import extract_frame, extract_run, load_boundary_payload

# Keep the contract independent of production constants, including literal False.
FLAGS = (
    "semantic_verified", "physical_field_boundary_verified",
    "maneuver_verified", "annotation_used_for_inference",
)
SURFACES = ("crop", "ground", "woody_vegetation", "water", "ditch")
OCCLUDERS = ("vehicle", "person", "tarp", "pole")
PASSED = "spatially_consistent"


def rectangle(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def make_evidence(shape, specifications=(), frame_id="synthetic-frame"):
    """Each specification is (id, semantic, mask[, spatial status])."""
    instances, masks = [], {}
    for item in specifications:
        iid, semantic, mask, *status = item
        key = "mask-" + iid
        instances.append({
            "instance_id": iid, "semantic": semantic, "mask_key": key,
            "spatial_check": {"status": status[0] if status else PASSED},
        })
        masks[key] = mask.copy()
    return {
        "frame_id": frame_id,
        "image": {"height": shape[0], "width": shape[1]},
        "instances": instances,
    }, masks


def split_evidence(shape=(16, 12), split=6, semantics=("crop", "ground")):
    h, w = shape
    return make_evidence(shape, [
        ("left", semantics[0], rectangle(shape, 0, h, 0, split)),
        ("right", semantics[1], rectangle(shape, 0, h, split, w)),
    ])


def edge_key(p, q):
    return tuple(sorted((tuple(p), tuple(q))))


def grid_contacts(labels):
    """Independent oracle: enumerate shared pixel faces, never trace a graph."""
    contacts = {}
    h, w = labels.shape
    for y in range(h):
        for x in range(w):
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx >= w or ny >= h:
                    continue
                a, b = int(labels[y, x]), int(labels[ny, nx])
                if a <= 0 or b <= 0 or a == b:
                    continue
                if dx:
                    p, q = (x + 0.5, y - 0.5), (x + 0.5, y + 0.5)
                else:
                    p, q = (x - 0.5, y + 0.5), (x + 0.5, y + 0.5)
                contacts[edge_key(p, q)] = ((y, x), (ny, nx))
    return contacts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def set_nested(payload, path, value):
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


class ContractAssertions(unittest.TestCase):
    def assert_flags_false(self, record):
        for flag in FLAGS:
            self.assertIn(flag, record)
            self.assertIs(record[flag], False, flag)

    def assert_geometry(self, frame, labels, evidence, masks, complete=True):
        """Check half-pixel faces, local positive support, and length conservation."""
        self.assert_flags_false(frame)
        self.assertEqual(frame["frame_id"], evidence["frame_id"])
        self.assertEqual(frame["image"], evidence["image"])
        self.assertEqual(labels.shape, (frame["image"]["height"], frame["image"]["width"]))
        contacts = grid_contacts(labels)
        seen, ids = set(), []
        for boundary in frame["boundaries"]:
            self.assert_flags_false(boundary)
            self.assertEqual(boundary["status"], "image_candidate")
            self.assertEqual(boundary["evidence_frame_id"], evidence["frame_id"])
            ids.append(boundary["boundary_id"])
            points = np.asarray(boundary["points_px"], dtype=float)
            self.assertEqual(points.ndim, 2)
            self.assertEqual(points.shape[1], 2)
            self.assertGreaterEqual(len(points), 2)
            self.assertTrue(np.isfinite(points).all())
            np.testing.assert_array_equal(points % 1, np.full_like(points, 0.5))
            directions = np.diff(points, axis=0)
            self.assertTrue(np.all(np.count_nonzero(directions, axis=1) == 1))
            self.assertTrue(np.all(np.abs(directions).sum(axis=1) > 0))
            for before, after in zip(directions, directions[1:]):
                self.assertNotEqual(before[0] * after[1] - before[1] * after[0], 0,
                                    "Only turns and endpoints should survive collinear compression")
            self.assertEqual(boundary["length_px"], float(np.abs(directions).sum()))
            self.assertEqual(boundary["closed"], bool(np.array_equal(points[0], points[-1])))
            if boundary["closed"]:
                self.assertEqual(boundary["endpoint_reasons"], [["closed"], ["closed"]])
            support_pixels = set()
            for start, delta in zip(points, directions):
                count = int(np.abs(delta).sum())
                step = np.sign(delta)
                for index in range(count):
                    key = edge_key(start + index * step, start + (index + 1) * step)
                    self.assertIn(key, contacts, "No diagonal, gap, conflict, occlusion, or FOV face is positive")
                    self.assertNotIn(key, seen, "A contact face must not be emitted twice")
                    seen.add(key)
                    pixels = contacts[key]
                    support_pixels.update(pixels)
                    pair = {SURFACES[int(labels[p]) - 1] for p in pixels}
                    self.assertEqual(pair, {boundary["side_a"]["semantic"], boundary["side_b"]["semantic"]})
            for side_name in ("side_a", "side_b"):
                side = boundary[side_name]
                expected_ids, expected_semantics = [], set()
                for instance in evidence["instances"]:
                    semantic = instance["semantic"]
                    primary = "ground" if semantic == "harvested_ground" else semantic
                    if instance["spatial_check"]["status"] != PASSED or primary != side["semantic"]:
                        continue
                    if any(labels[p] == SURFACES.index(primary) + 1 and masks[instance["mask_key"]][p]
                           for p in support_pixels):
                        expected_ids.append(instance["instance_id"])
                        expected_semantics.add(semantic)
                self.assertEqual(side["instance_ids"], sorted(expected_ids))
                self.assertEqual(side["source_semantics"], sorted(expected_semantics))
                self.assertTrue(expected_ids)
        self.assertEqual(len(ids), len(set(ids)))
        if complete:
            self.assertEqual(seen, set(contacts))
        self.assertEqual(frame["stats"]["boundary_count"], len(ids))
        self.assertEqual(frame["stats"]["boundary_length_px"], len(seen))
        audit = frame["stats"]["contact_audit"]
        self.assertEqual(sum(a["raw_length_px"] for a in audit), len(contacts))
        self.assertEqual(frame["stats"]["short_segments"], sum(a["short_segments"] for a in audit))
        self.assertEqual(frame["stats"]["short_length_px"], sum(a["short_length_px"] for a in audit))
        self.assertEqual(len(seen) + frame["stats"]["short_length_px"], len(contacts))

    def extract_checked(self, evidence, masks, min_length_px=0):
        frame, labels = extract_frame(evidence, masks, min_length_px=min_length_px)
        self.assert_geometry(frame, labels, evidence, masks, complete=min_length_px == 0)
        return frame, labels

    def assert_line(self, boundary, expected):
        # Orientation is not a semantic assertion.
        actual = boundary["points_px"]
        self.assertTrue(actual == expected or actual == expected[::-1], (actual, expected))


class ExtractFrameTests(ContractAssertions):
    def test_empty_input_is_unknown_not_ground_or_boundary(self):
        for specs in ([], [("empty", "crop", np.zeros((16, 12), bool))]):
            with self.subTest(instances=len(specs)):
                evidence, masks = make_evidence((16, 12), specs)
                frame, labels = self.extract_checked(evidence, masks)
                np.testing.assert_array_equal(labels, np.zeros((16, 12), dtype=np.int8))
                self.assertEqual(frame["boundaries"], [])
                self.assertEqual(frame["diagnostics"], [])
                self.assertEqual(frame["status"], "no_supported_interface")
                self.assertEqual(frame["stats"]["label_pixels"]["unknown"], 192)
                self.assertEqual(frame["stats"]["contact_audit"], [])

    def test_crop_ground_straight_exact_coordinates_length_and_support(self):
        for horizontal in (False, True):
            with self.subTest(horizontal=horizontal):
                evidence, masks = split_evidence()
                if horizontal:
                    evidence["image"] = {"height": 12, "width": 16}
                    masks = {key: mask.T.copy() for key, mask in masks.items()}
                frame, labels = extract_frame(evidence, masks)  # Exercise the default cutoff.
                self.assert_geometry(frame, labels, evidence, masks)
                expected_labels = np.repeat(np.array([[1] * 6 + [2] * 6]), 16, axis=0)
                np.testing.assert_array_equal(labels, expected_labels.T if horizontal else expected_labels)
                self.assertEqual(len(frame["boundaries"]), 1)
                boundary = frame["boundaries"][0]
                expected = [[-0.5, 5.5], [15.5, 5.5]] if horizontal else [[5.5, -0.5], [5.5, 15.5]]
                self.assert_line(boundary, expected)
                self.assertEqual(boundary["length_px"], 16.0)
                self.assertEqual(boundary["type"], "crop_ground_interface")
                self.assertFalse(boundary["closed"])
                self.assertEqual(boundary["endpoint_reasons"], [["fov"], ["fov"]])
                self.assertEqual(boundary["side_a"]["instance_ids"], ["left"])
                self.assertEqual(boundary["side_b"]["instance_ids"], ["right"])

    def test_unknown_strip_between_surfaces_is_not_bridged(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 5)),
            ("g", "ground", rectangle(shape, 0, 16, 6, 12)),
        ])
        frame, labels = self.extract_checked(evidence, masks)
        self.assertTrue(np.all(labels[:, 5] == 0))
        self.assertEqual(frame["boundaries"], [])
        self.assertEqual(frame["stats"]["contact_audit"], [])
        unknown = {d["semantic"]: d["length_px"] for d in frame["diagnostics"] if d["reason"] == "unknown"}
        self.assertEqual(unknown, {"crop": 16.0, "ground": 16.0})

    def test_one_sided_crop_contour_preserves_unknown_side_and_local_support(self):
        shape = (16, 12)
        crop = rectangle(shape, 0, 16, 0, 5)
        evidence, masks = make_evidence(shape, [("c", "crop", crop)])
        frame, _ = self.extract_checked(evidence, masks)
        self.assertEqual(len(frame["coverage_contours"]), 1)
        contour = frame["coverage_contours"][0]
        self.assertEqual(contour["boundary_id"], "C001")
        self.assertEqual(contour["status"], "one_sided_candidate")
        self.assertEqual(contour["type"], "crop_coverage_contour")
        self.assertEqual(contour["side_a"]["instance_ids"], ["c"])
        self.assertEqual(contour["side_b"], {"semantic": "unknown", "instance_ids": [], "source_semantics": []})
        self.assert_line(contour, [[4.5, -0.5], [4.5, 15.5]])
        self.assertEqual(contour["length_px"], 16)
        self.assert_flags_false(contour)
        self.assertEqual(frame["stats"]["coverage_count"], 1)
        self.assertFalse(frame["boundaries"])

    def test_coverage_contours_do_not_bridge_occluders_or_fov_edges(self):
        shape = (20, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 20, 0, 5)),
            ("v", "vehicle", rectangle(shape, 8, 12, 0, 12)),
        ])
        frame, _ = self.extract_checked(evidence, masks)
        contours = frame["coverage_contours"]
        self.assertEqual(len(contours), 2)
        self.assertEqual([c["length_px"] for c in contours], [8, 8])
        self.assertTrue(all(set(p[0] for p in c["points_px"]) == {4.5} for c in contours))
        self.assertTrue(all("occlusion" in sum(c["endpoint_reasons"], []) for c in contours))
        short, _ = extract_frame(evidence, masks, min_length_px=12)
        self.assertFalse(short["coverage_contours"])
        self.assertEqual(short["stats"]["short_coverage_count"], 2)
        self.assertEqual(short["stats"]["short_coverage_length_px"], 16)

    def test_coverage_contours_do_not_convert_conflict_to_unknown(self):
        shape = (16, 12)
        crop = rectangle(shape, 0, 16, 0, 6)
        ground = rectangle(shape, 0, 16, 5, 12)
        evidence, masks = make_evidence(shape, [("c", "crop", crop), ("g", "ground", ground)])
        frame, _ = self.extract_checked(evidence, masks)
        self.assertFalse(frame["coverage_contours"])
        self.assertFalse(frame["boundaries"])
        self.assertTrue(any(d["reason"] == "conflict" for d in frame["diagnostics"]))

    def test_diagonal_contact_is_not_four_neighbor_contact(self):
        shape = (3, 3)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 1, 0, 1)),
            ("g", "ground", rectangle(shape, 1, 2, 1, 2)),
        ])
        frame, _ = self.extract_checked(evidence, masks)
        self.assertEqual(frame["boundaries"], [])
        self.assertEqual(frame["stats"]["contact_audit"], [])

    def test_unknown_conflict_and_each_occluder_interrupt_contact(self):
        shape = (18, 12)
        for semantic in (None, "woody_vegetation", *OCCLUDERS):
            with self.subTest(interruption=semantic):
                evidence, masks = split_evidence(shape)
                reason, value = "unknown", 0
                if semantic is None:
                    for mask in masks.values():
                        mask[6:9] = False
                else:
                    extra, extra_masks = make_evidence(shape, [
                        ("block", semantic, rectangle(shape, 6, 9, 0, 12)),
                    ])
                    evidence["instances"].extend(extra["instances"])
                    masks.update(extra_masks)
                    reason, value = ("conflict", -1) if semantic == "woody_vegetation" else ("occlusion", -2)
                frame, labels = self.extract_checked(evidence, masks)
                expected = np.repeat(np.array([[1] * 6 + [2] * 6]), 18, axis=0)
                expected[6:9] = value
                np.testing.assert_array_equal(labels, expected)
                boundaries = sorted(frame["boundaries"], key=lambda b: min(p[1] for p in b["points_px"]))
                self.assertEqual(len(boundaries), 2)
                for boundary, points, length in zip(boundaries,
                        ([[5.5, -0.5], [5.5, 5.5]], [[5.5, 8.5], [5.5, 17.5]]), (6, 9)):
                    self.assert_line(boundary, points)
                    self.assertEqual(boundary["length_px"], length)
                    self.assertFalse(boundary["closed"])
                    endpoint_map = dict(zip(map(tuple, boundary["points_px"]), boundary["endpoint_reasons"]))
                    for point, reasons in endpoint_map.items():
                        self.assertEqual(reasons, ["fov"] if point[1] in (-0.5, 17.5) else [reason])
                self.assertEqual(frame["stats"]["label_pixels"][reason], 36)
                diagnostics = {d["semantic"]: d["length_px"] for d in frame["diagnostics"] if d["reason"] == reason}
                self.assertEqual(diagnostics, {"crop": 12.0, "ground": 12.0})

    def test_occlusion_takes_precedence_over_incompatible_overlap(self):
        full = np.ones((4, 5), dtype=bool)
        evidence, masks = make_evidence(full.shape, [
            ("c", "crop", full), ("g", "ground", full), ("v", "vehicle", full),
        ])
        frame, labels = self.extract_checked(evidence, masks)
        self.assertTrue(np.all(labels == -2))
        self.assertEqual(frame["stats"]["label_pixels"]["conflict"], 0)
        self.assertEqual(frame["stats"]["label_pixels"]["occlusion"], 20)
        self.assertEqual(frame["boundaries"], [])

    def test_each_incompatible_surface_overlap_is_unknown_not_positive(self):
        full = np.ones((4, 5), dtype=bool)
        for a, b in combinations(SURFACES, 2):
            with self.subTest(pair=(a, b)):
                evidence, masks = make_evidence(full.shape, [("a", a, full), ("b", b, full)])
                frame, labels = self.extract_checked(evidence, masks)
                self.assertTrue(np.all(labels == -1))
                self.assertEqual(frame["boundaries"], [])
                self.assertEqual(frame["stats"]["label_pixels"]["conflict"], 20)

    def test_supported_hole_closes_but_unknown_hole_has_only_diagnostic(self):
        shape = (9, 9)
        hole = rectangle(shape, 2, 7, 3, 6)
        for supported in (False, True):
            with self.subTest(supported=supported):
                specs = [("outer", "crop", ~hole)]
                if supported:
                    specs.append(("inner", "ground", hole))
                evidence, masks = make_evidence(shape, specs)
                frame, _ = self.extract_checked(evidence, masks)
                if not supported:
                    self.assertEqual(frame["boundaries"], [])
                    self.assertEqual([d["length_px"] for d in frame["diagnostics"] if d["reason"] == "unknown"], [16.0])
                else:
                    self.assertEqual(len(frame["boundaries"]), 1)
                    boundary = frame["boundaries"][0]
                    self.assertTrue(boundary["closed"])
                    self.assertEqual(boundary["length_px"], 16.0)
                    self.assertEqual(len(boundary["points_px"]), 5)
                    self.assertEqual(set(map(tuple, boundary["points_px"])),
                                     {(2.5, 1.5), (5.5, 1.5), (5.5, 6.5), (2.5, 6.5)})

    def test_one_pixel_unknown_break_in_hole_is_not_forced_closed(self):
        shape = (9, 9)
        hole = rectangle(shape, 2, 7, 3, 6)
        inner = hole.copy()
        inner[2, 4] = False
        evidence, masks = make_evidence(shape, [("outer", "crop", ~hole), ("inner", "ground", inner)])
        frame, labels = self.extract_checked(evidence, masks)
        self.assertEqual(labels[2, 4], 0)
        self.assertEqual(len(frame["boundaries"]), 1)
        boundary = frame["boundaries"][0]
        self.assertFalse(boundary["closed"])
        self.assertEqual(boundary["length_px"], 15.0)
        self.assertEqual({tuple(boundary["points_px"][0]), tuple(boundary["points_px"][-1])},
                         {(3.5, 1.5), (4.5, 1.5)})
        self.assertEqual(boundary["endpoint_reasons"], [["unknown"], ["unknown"]])

    def test_staircase_preserves_every_turn_and_only_compresses_collinear_faces(self):
        shape = (9, 10)
        crop = np.zeros(shape, bool)
        for rows, split in ((slice(0, 3), 3), (slice(3, 6), 5), (slice(6, 9), 4)):
            crop[rows, :split] = True
        evidence, masks = make_evidence(shape, [("c", "crop", crop), ("g", "ground", ~crop)])
        frame, labels = extract_frame(evidence, masks)
        self.assert_geometry(frame, labels, evidence, masks)
        self.assertEqual(len(frame["boundaries"]), 1)
        self.assert_line(frame["boundaries"][0], [
            [2.5, -0.5], [2.5, 2.5], [4.5, 2.5], [4.5, 5.5], [3.5, 5.5], [3.5, 8.5],
        ])
        self.assertEqual(frame["boundaries"][0]["length_px"], 12.0)

    def test_image_border_is_fov_diagnostic_not_boundary(self):
        shape = (5, 7)
        evidence, masks = make_evidence(shape, [("c", "crop", np.ones(shape, bool))])
        frame, _ = self.extract_checked(evidence, masks)
        self.assertEqual(frame["boundaries"], [])
        self.assertEqual(frame["stats"]["contact_audit"], [])
        self.assertEqual(len(frame["diagnostics"]), 1)
        diagnostic = frame["diagnostics"][0]
        self.assertEqual(diagnostic["reason"], "fov")
        self.assertEqual(diagnostic["semantic"], "crop")
        self.assertEqual(diagnostic["length_px"], 24.0)
        self.assertEqual(len(diagnostic["polylines_px"]), 1)
        line = diagnostic["polylines_px"][0]
        self.assertEqual(line[0], line[-1])
        self.assertEqual(len(line), 5)
        self.assertEqual(set(map(tuple, line)), {(-0.5, -0.5), (6.5, -0.5), (6.5, 4.5), (-0.5, 4.5)})

    def test_ground_and_harvested_overlap_union_preserves_source_subtypes(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 6)),
            ("g", "ground", rectangle(shape, 0, 12, 6, 12)),
            ("h", "harvested_ground", rectangle(shape, 4, 16, 6, 12)),
        ])
        frame, labels = self.extract_checked(evidence, masks)
        self.assertTrue(np.all(labels[:, 6:] == 2))
        self.assertEqual(frame["stats"]["label_pixels"]["conflict"], 0)
        self.assertEqual(len(frame["boundaries"]), 1)
        self.assertEqual(frame["boundaries"][0]["side_b"], {
            "semantic": "ground", "instance_ids": ["g", "h"],
            "source_semantics": ["ground", "harvested_ground"],
        })
        self.assertEqual(frame["boundaries"][0]["length_px"], 16.0)
        evidence["instances"] = evidence["instances"][1:]
        frame, _ = self.extract_checked(evidence, masks)
        self.assertEqual(frame["boundaries"], [])  # No ground/harvested subtype interface.

    def test_same_semantic_instances_union_once_with_only_local_support(self):
        shape = (24, 12)
        evidence, masks = make_evidence(shape, [
            ("c-top-a", "crop", rectangle(shape, 0, 4, 0, 6)),
            ("c-top-b", "crop", rectangle(shape, 2, 6, 0, 6)),
            ("c-bottom", "crop", rectangle(shape, 10, 18, 0, 6)),
            ("c-remote", "crop", rectangle(shape, 20, 24, 0, 3)),
            ("g-top-a", "ground", rectangle(shape, 0, 5, 6, 12)),
            ("g-top-b", "ground", rectangle(shape, 1, 6, 6, 12)),
            ("g-bottom", "ground", rectangle(shape, 10, 18, 6, 12)),
            ("g-remote", "ground", rectangle(shape, 20, 24, 9, 12)),
        ])
        # High-scoring remote instances must not become geometry representatives.
        for instance in evidence["instances"]:
            instance["score"] = 0.99 if "remote" in instance["instance_id"] else 0.1
        frame, labels = self.extract_checked(evidence, masks)
        self.assertEqual(frame["stats"]["label_pixels"]["conflict"], 0)
        self.assertEqual(frame["stats"]["used_instance_ids"], sorted(i["instance_id"] for i in evidence["instances"]))
        boundaries = sorted(frame["boundaries"], key=lambda b: b["length_px"])
        self.assertEqual([b["length_px"] for b in boundaries], [6.0, 8.0])
        for boundary, suffixes in zip(boundaries, (["top-a", "top-b"], ["bottom"])):
            self.assertEqual(boundary["side_a"]["instance_ids"], ["c-" + s for s in suffixes])
            self.assertEqual(boundary["side_b"]["instance_ids"], ["g-" + s for s in suffixes])
        reordered = deepcopy(evidence)
        reordered["instances"].reverse()
        again, again_labels = extract_frame(reordered, masks, min_length_px=0)
        self.assertEqual(again, frame)
        np.testing.assert_array_equal(again_labels, labels)

    def test_rejected_surface_masks_are_not_positive_evidence(self):
        for status in ("outside_proposal", "rejected", "unchecked"):
            with self.subTest(status=status):
                evidence, masks = split_evidence()
                evidence["instances"][1]["spatial_check"]["status"] = status
                frame, labels = self.extract_checked(evidence, masks)
                self.assertTrue(np.all(labels[:, 6:] == 0))
                self.assertEqual(frame["boundaries"], [])
                self.assertEqual(frame["stats"]["used_instance_ids"], ["left"])
                self.assertEqual(frame["stats"]["rejected_instance_ids"], ["right"])
                self.assertTrue(any(d["reason"] == "unknown" for d in frame["diagnostics"]))

    def test_rejected_overlaps_and_occluders_do_not_erase_passed_contact(self):
        for semantic in ("woody_vegetation", *OCCLUDERS):
            with self.subTest(semantic=semantic):
                evidence, masks = split_evidence()
                extra, arrays = make_evidence((16, 12), [
                    ("rejected", semantic, np.ones((16, 12), bool), "outside_proposal"),
                ])
                evidence["instances"].extend(extra["instances"])
                masks.update(arrays)
                frame, labels = self.extract_checked(evidence, masks)
                self.assertTrue(np.all(labels > 0))
                self.assertEqual(frame["stats"]["boundary_length_px"], 16.0)
                self.assertEqual(frame["stats"]["rejected_instance_ids"], ["rejected"])

    def test_unsupported_semantic_does_not_supply_or_block_surface(self):
        evidence, masks = split_evidence()
        extra, arrays = make_evidence((16, 12), [("other", "sky", np.ones((16, 12), bool))])
        evidence["instances"].extend(extra["instances"])
        masks.update(arrays)
        frame, _ = self.extract_checked(evidence, masks)
        self.assertEqual(frame["stats"]["unsupported_instance_ids"], ["other"])
        self.assertEqual(frame["stats"]["used_instance_ids"], ["left", "right"])
        self.assertEqual(frame["stats"]["boundary_length_px"], 16.0)

    def test_all_surface_pairs_are_visible_candidates_not_near_side_or_turn_claims(self):
        for a, b in combinations(SURFACES, 2):
            with self.subTest(pair=(a, b)):
                evidence, masks = split_evidence(semantics=(a, b))
                frame, labels = self.extract_checked(evidence, masks)
                self.assertTrue(np.all(labels > 0))  # In particular, ditch is not an occluder.
                self.assertEqual(len(frame["boundaries"]), 1)
                boundary = frame["boundaries"][0]
                expected_type = ("ditch_visible_interface" if "ditch" in (a, b) else
                                 "water_visible_interface" if "water" in (a, b) else
                                 "vegetation_coverage_interface" if "woody_vegetation" in (a, b) else
                                 "crop_ground_interface")
                self.assertEqual(boundary["type"], expected_type)
                self.assertEqual(boundary["length_px"], 16.0)
                for claim in ("near_side", "far_side", "turn_allowed", "headland", "maneuver_allowed"):
                    self.assertNotIn(claim, boundary)
                self.assert_flags_false(boundary)

    def test_default_short_segment_is_removed_but_fully_audited(self):
        evidence, masks = split_evidence(shape=(10, 12))
        frame, labels = extract_frame(evidence, masks)
        self.assert_geometry(frame, labels, evidence, masks, complete=False)
        self.assertEqual(frame["boundaries"], [])
        self.assertEqual(frame["status"], "no_supported_interface")
        self.assertEqual(frame["stats"]["contact_audit"], [{
            "side_a": "crop", "side_b": "ground", "raw_segments": 1, "raw_length_px": 10,
            "short_segments": 1, "short_length_px": 10,
        }])
        self.assertEqual(frame["stats"]["short_segments"], 1)
        self.assertEqual(frame["stats"]["short_length_px"], 10)
        self.assertEqual(frame["stats"]["boundary_length_px"], 0)

    def test_cutoff_is_inclusive_and_mixed_short_contacts_conserve_length(self):
        evidence, masks = split_evidence(shape=(18, 12))
        for mask in masks.values():
            mask[3:5] = False
            mask[17:] = False
        for cutoff, lengths, short_count, short_length in ((0, [3, 12], 0, 0), (12, [12], 1, 3), (12.5, [], 2, 15)):
            with self.subTest(cutoff=cutoff):
                frame, _ = self.extract_checked(evidence, masks, cutoff)
                self.assertEqual(sorted(b["length_px"] for b in frame["boundaries"]), lengths)
                self.assertEqual(frame["stats"]["contact_audit"], [{
                    "side_a": "crop", "side_b": "ground", "raw_segments": 2, "raw_length_px": 15,
                    "short_segments": short_count, "short_length_px": short_length,
                }])

    def test_checkerboard_four_way_junctions_split_without_losing_or_reusing_edges(self):
        for size in (2, 4):
            with self.subTest(size=size):
                crop = np.indices((size, size)).sum(axis=0) % 2 == 0
                evidence, masks = make_evidence(crop.shape, [("c", "crop", crop), ("g", "ground", ~crop)])
                frame, _ = self.extract_checked(evidence, masks)
                expected_count = 2 * size * (size - 1)
                self.assertEqual(len(frame["boundaries"]), expected_count)
                self.assertEqual([b["length_px"] for b in frame["boundaries"]], [1.0] * expected_count)
                self.assertEqual(frame["stats"]["boundary_length_px"], expected_count)
                self.assertEqual(frame["stats"]["contact_audit"][0]["raw_segments"], expected_count)
                if size == 2:
                    ends = Counter(tuple(p) for b in frame["boundaries"] for p in b["points_px"])
                    self.assertEqual(ends, Counter({(0.5, 0.5): 4, (-0.5, 0.5): 1, (1.5, 0.5): 1,
                                                   (0.5, -0.5): 1, (0.5, 1.5): 1}))
                    for boundary in frame["boundaries"]:
                        self.assertFalse(boundary["closed"])
                        for point, reasons in zip(boundary["points_px"], boundary["endpoint_reasons"]):
                            self.assertEqual(reasons, ["junction"] if point == [0.5, 0.5] else ["fov"])

    def test_mask_wrong_shape_or_nonboolean_dtype_is_rejected_even_if_instance_rejected(self):
        invalid = [
            np.ones((15, 12), bool), np.ones((12, 16), bool), np.ones((16, 12, 1), bool),
            np.array(True), np.ones((16, 12), np.uint8), np.ones((16, 12), np.int32),
            np.ones((16, 12), np.float32), np.full((16, 12), True, dtype=object),
        ]
        for mask in invalid:
            for status in (PASSED, "outside_proposal"):
                with self.subTest(shape=mask.shape, dtype=str(mask.dtype), status=status):
                    evidence, masks = make_evidence((16, 12), [("bad", "crop", mask, status)])
                    with self.assertRaisesRegex(ValueError, "boolean.*dimensions"):
                        extract_frame(evidence, masks)

    def test_invalid_dimensions_cutoff_and_duplicate_instance_identity_are_rejected(self):
        evidence, masks = split_evidence()
        for cutoff in (-1, math.inf, -math.inf, math.nan):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                extract_frame(evidence, masks, min_length_px=cutoff)
        for dimension in ("width", "height"):
            for value in (0, -1, True, 12.0):
                changed = deepcopy(evidence)
                changed["image"][dimension] = value
                with self.subTest(dimension=dimension, value=value), self.assertRaises(ValueError):
                    extract_frame(changed, masks)
        changed = deepcopy(evidence)
        changed["instances"][1]["instance_id"] = changed["instances"][0]["instance_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate instance"):
            extract_frame(changed, masks)


class GapClosureFrameTests(ContractAssertions):
    def test_two_pixel_band_closes_only_at_threshold_with_full_audit(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 5)),
            ("g", "ground", rectangle(shape, 0, 16, 7, 12)),
        ])
        for gap_px in (0, 0.5):
            with self.subTest(gap_px=gap_px):
                frame, labels = extract_frame(evidence, masks, min_length_px=0, gap_closure_px=gap_px)
                self.assertEqual(frame["boundaries"], [])
                self.assertEqual(frame["gap_closed_boundaries"], [])
                self.assertTrue(np.all(labels[:, 5:7] == 0))
                self.assertEqual(frame["stats"]["gap_closure"], {
                    "assigned_pixels": 0, "by_pair": {}, "by_class": {},
                    "gap_closed_count": 0, "gap_closed_length_px": 0,
                    "short_gap_closed_segments": 0, "short_gap_closed_length_px": 0})
        frame, labels = extract_frame(evidence, masks, min_length_px=0, gap_closure_px=1)
        self.assertEqual(frame["boundaries"], [])
        self.assertEqual(len(frame["gap_closed_boundaries"]), 1)
        record = frame["gap_closed_boundaries"][0]
        self.assert_flags_false(record)
        self.assertEqual(record["boundary_id"], "GC001")
        self.assertEqual(record["status"], "gap_closed_candidate")
        self.assertEqual(record["type"], "crop_ground_interface")
        self.assertEqual(record["evidence_frame_id"], evidence["frame_id"])
        self.assertEqual(record["side_a"], {"semantic": "crop", "instance_ids": ["c"], "source_semantics": ["crop"]})
        self.assertEqual(record["side_b"], {"semantic": "ground", "instance_ids": ["g"], "source_semantics": ["ground"]})
        self.assert_line(record, [[5.5, -0.5], [5.5, 15.5]])
        self.assertEqual(record["length_px"], 16.0)
        self.assertFalse(record["closed"])
        self.assertEqual(record["endpoint_reasons"], [["fov"], ["fov"]])
        self.assertEqual(record["gap"], {"assigned_neighbor_px": 32, "side_a_assigned_px": 16,
                                         "side_b_assigned_px": 16,
                                         "width_px": {"min": 3.0, "mean": 3.0, "max": 3.0}})
        self.assertEqual(frame["stats"]["gap_closure"], {
            "assigned_pixels": 32, "by_pair": {"crop|ground": 32}, "by_class": {"crop": 16, "ground": 16},
            "gap_closed_count": 1, "gap_closed_length_px": 16.0,
            "short_gap_closed_segments": 0, "short_gap_closed_length_px": 0})

    def test_wider_band_requires_matching_threshold(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 5)),
            ("g", "ground", rectangle(shape, 0, 16, 9, 12)),
        ])
        for gap_px, count in ((2, 0), (3, 1)):
            with self.subTest(gap_px=gap_px):
                frame, labels = extract_frame(evidence, masks, min_length_px=0, gap_closure_px=gap_px)
                self.assertEqual(len(frame["gap_closed_boundaries"]), count)
                if count:
                    record = frame["gap_closed_boundaries"][0]
                    self.assert_line(record, [[6.5, -0.5], [6.5, 15.5]])
                    self.assertEqual(record["gap"]["width_px"], {"min": 5.0, "mean": 5.0, "max": 5.0})

    def test_closure_does_not_touch_observed_boundaries_or_diagnostics(self):
        evidence, masks = split_evidence()
        baseline, baseline_labels = extract_frame(evidence, masks)
        closed, closed_labels = extract_frame(evidence, masks, gap_closure_px=5)
        for key in ("boundaries", "coverage_contours", "diagnostics", "status"):
            self.assertEqual(closed[key], baseline[key])
        self.assertEqual(closed["gap_closed_boundaries"], [])
        self.assertEqual(closed["stats"]["gap_closure"]["assigned_pixels"], 0)
        np.testing.assert_array_equal(baseline_labels, closed_labels)

    def test_woody_vegetation_band_is_never_closed(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 5)),
            ("w", "woody_vegetation", rectangle(shape, 0, 16, 7, 12)),
        ])
        frame, labels = extract_frame(evidence, masks, gap_closure_px=10)
        self.assertEqual(frame["gap_closed_boundaries"], [])
        self.assertEqual(frame["stats"]["gap_closure"]["assigned_pixels"], 0)
        self.assertTrue(np.all(labels[:, 5:7] == 0))

    def test_occluded_band_is_never_reassigned(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 5)),
            ("g", "ground", rectangle(shape, 0, 16, 7, 12)),
            ("v", "vehicle", rectangle(shape, 0, 16, 5, 7)),
        ])
        frame, labels = extract_frame(evidence, masks, gap_closure_px=10)
        self.assertEqual(frame["gap_closed_boundaries"], [])
        self.assertEqual(frame["stats"]["gap_closure"]["assigned_pixels"], 0)

    def test_conflict_band_blocks_closed_interface_even_when_flanks_assign(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 16, 0, 5)),
            ("g", "ground", rectangle(shape, 0, 16, 6, 12)),
            ("c2", "crop", rectangle(shape, 0, 16, 6, 9)),
        ])
        frame, labels = extract_frame(evidence, masks, gap_closure_px=3)
        stats = frame["stats"]["gap_closure"]
        self.assertEqual(stats["assigned_pixels"], 16)
        self.assertEqual(stats["by_pair"], {"crop|ground": 16})
        self.assertEqual(stats["by_class"], {"crop": 16})
        self.assertEqual(frame["gap_closed_boundaries"], [])

    def test_short_gap_closed_segments_are_counted_not_emitted(self):
        shape = (16, 12)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 0, 6, 0, 5)),
            ("g", "ground", rectangle(shape, 0, 6, 7, 12)),
        ])
        frame, _ = extract_frame(evidence, masks, min_length_px=12, gap_closure_px=1)
        self.assertEqual(frame["gap_closed_boundaries"], [])
        stats = frame["stats"]["gap_closure"]
        self.assertEqual(stats["short_gap_closed_segments"], 1)
        self.assertEqual(stats["short_gap_closed_length_px"], 6.0)
        self.assertEqual(stats["assigned_pixels"], 12)

    def test_support_attributes_the_true_nearest_mask_pixel(self):
        # Offset thin strips: a misdecoded nearest-pixel map lands on rows/cols
        # outside the strips, so both sides would lose their supporting instance.
        shape = (12, 16)
        evidence, masks = make_evidence(shape, [
            ("c", "crop", rectangle(shape, 2, 10, 0, 1)),
            ("g", "ground", rectangle(shape, 2, 10, 15, 16)),
        ])
        frame, _ = extract_frame(evidence, masks, min_length_px=0, gap_closure_px=20)
        self.assertEqual(len(frame["gap_closed_boundaries"]), 1)
        record = frame["gap_closed_boundaries"][0]
        self.assert_line(record, [[7.5, -0.5], [7.5, 11.5]])
        self.assertEqual(record["side_a"]["instance_ids"], ["c"])
        self.assertEqual(record["side_b"]["instance_ids"], ["g"])

    def test_invalid_gap_closure_parameter_is_rejected(self):
        evidence, masks = make_evidence((16, 12))
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                extract_frame(evidence, masks, gap_closure_px=bad)


def write_synthetic_run(base):
    """Real PNG, JSON, boolean NPZ, source fields, and independently computed hashes."""
    run = base / "source-run"
    run.mkdir()
    entries = []
    for frame_id, frame_index, timestamp, split in (
            ("late", 33, 2.75, 4), ("empty", 54, 4.5, None), ("early", 3, 0.25, 6)):
        folder = run / frame_id
        folder.mkdir()
        source = {
            "video_name": "synthetic.mp4", "video_sha256": hashlib.sha256(b"synthetic video").hexdigest(),
            "frame_index": frame_index, "capture_time_s": timestamp,
        }
        evidence, masks = make_evidence((16, 12)) if split is None else split_evidence(split=split)
        evidence.update({"frame_id": frame_id, "source": source})
        image = np.full((16, 12, 3), frame_index, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise AssertionError("OpenCV could not encode the synthetic image")
        (folder / "frame.png").write_bytes(encoded.tobytes())
        write_json(folder / "evidence.json", evidence)
        np.savez_compressed(folder / "masks.npz", **masks)
        entry = {"frame_id": frame_id, **source,
                 "image_path": f"{frame_id}/frame.png", "evidence_path": f"{frame_id}/evidence.json",
                 "masks_path": f"{frame_id}/masks.npz"}
        entry["artifact_sha256"] = {entry[key]: digest(run / entry[key])
                                    for key in ("image_path", "evidence_path", "masks_path")}
        entries.append(entry)
    manifest = {"status": "completed", "frames": entries}
    write_json(run / "manifest.json", manifest)
    return run, manifest


class ExtractRunAndPayloadTests(ContractAssertions):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="visible-boundaries-test-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.run, self.manifest = write_synthetic_run(self.base)
        self.payload = extract_run(self.run)
        self.output = self.base / "boundaries.json"
        write_json(self.output, self.payload)
        self.entry = self.manifest["frames"][0]
        self.evidence = json.loads((self.run / self.entry["evidence_path"]).read_bytes())

    def load_changed(self, payload):
        write_json(self.output, payload)
        return load_boundary_payload(self.output, self.run)

    def rewrite_evidence_with_valid_hash(self, evidence):
        path = self.run / self.entry["evidence_path"]
        write_json(path, evidence)
        manifest = deepcopy(self.manifest)
        manifest["frames"][0]["artifact_sha256"][self.entry["evidence_path"]] = digest(path)
        write_json(self.run / "manifest.json", manifest)

    def test_extract_run_and_load_round_trip_bind_sources_without_temporal_fill(self):
        self.assertEqual(load_boundary_payload(self.output, self.run), self.payload)
        self.assertEqual(self.payload["schema_version"], "visible_boundaries_v1")
        self.assertEqual(self.payload["run"], self.run.name)
        self.assertEqual(self.payload["source_manifest_sha256"], digest(self.run / "manifest.json"))
        self.assert_flags_false(self.payload)
        self.assertEqual([f["frame_id"] for f in self.payload["frames"]], ["early", "late", "empty"])
        self.assertEqual([f["capture_time_s"] for f in self.payload["frames"]], [0.25, 2.75, 4.5])
        self.assertEqual(self.payload["summary"], {
            "frames": 3, "frames_with_boundaries": 2, "boundary_count": 2,
            "boundary_length_px": 32.0, "coverage_count": 0, "frames_with_coverage": 0,
            "gap_closed_count": 0, "frames_with_gap_closed": 0, "gap_closed_length_px": 0,
            "types": {"crop_ground_interface": 2},
        })
        self.assertEqual(self.payload["parameters"], {
            "min_length_px": 12, "connectivity": 4, "gap_bridging_px": 0, "gap_closure_px": 0,
            "geometry_simplification": "collinear_only",
            "surface_class": {**{s: s for s in SURFACES}, "harvested_ground": "ground"},
            "occluders": list(OCCLUDERS), "instance_policy": "union_of_spatially_consistent_masks",
            "temporal_identity_used": False,
        })
        self.assertEqual(self.payload["implementation_sha256"], {
            "extract_visible_boundaries.py": digest(PKG / "evidence" / "extract_visible_boundaries.py"),
            "semantic_pipeline.py": digest(PKG / "semantic_pipeline.py"),
        })
        entries = {e["frame_id"]: e for e in self.manifest["frames"]}
        for frame in self.payload["frames"]:
            entry = entries[frame["frame_id"]]
            self.assert_flags_false(frame)
            self.assertEqual(frame["source"], {
                "video_name": entry["video_name"], "video_sha256": entry["video_sha256"],
                "frame_index": entry["frame_index"], "artifact_sha256": entry["artifact_sha256"],
            })
            evidence = json.loads((self.run / entry["evidence_path"]).read_bytes())
            with np.load(self.run / entry["masks_path"], allow_pickle=False) as masks:
                direct, labels = extract_frame(evidence, masks)
                self.assert_geometry(frame, labels, evidence, masks)
                self.assertEqual(frame["boundaries"], direct["boundaries"])
                self.assertEqual(frame["diagnostics"], direct["diagnostics"])
            for boundary in frame["boundaries"]:
                self.assertNotIn("track_id", boundary)
        self.assertEqual(self.payload["frames"][0]["boundaries"][0]["boundary_id"], "B001")
        self.assertEqual(self.payload["frames"][1]["boundaries"][0]["boundary_id"], "B001")
        self.assertEqual(self.payload["frames"][2]["boundaries"], [])
        self.assertIn("ditch near/far side is unknown", " ".join(self.payload["notes"]))

    def test_pre_gap_closure_payload_format_still_loads(self):
        legacy = deepcopy(self.payload)
        legacy["parameters"].pop("gap_closure_px")
        legacy["summary"].pop("gap_closed_count")
        legacy["summary"].pop("frames_with_gap_closed")
        legacy["summary"].pop("gap_closed_length_px")
        for frame in legacy["frames"]:
            frame.pop("gap_closed_boundaries")
            frame["stats"].pop("gap_closure")
        self.assertEqual(self.load_changed(legacy), legacy)

    def test_each_source_artifact_hash_tamper_is_rejected_by_extractor_and_loader(self):
        for field in ("image_path", "evidence_path", "masks_path"):
            with self.subTest(field=field):
                path = self.run / self.entry[field]
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"tampered")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        extract_run(self.run)
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        load_boundary_payload(self.output, self.run)
                finally:
                    path.write_bytes(original)

    def test_manifest_byte_change_invalidates_existing_payload(self):
        path = self.run / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "does not match source run"):
            load_boundary_payload(self.output, self.run)

    def test_source_evidence_identity_mismatches_rejected_after_rehash(self):
        mutations = [
            (("frame_id",), "different-frame"), (("source", "video_name"), "other.mp4"),
            (("source", "video_sha256"), "0" * 64), (("source", "frame_index"), 999),
            (("source", "capture_time_s"), 99.0), (("source",), {}),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                evidence = deepcopy(self.evidence)
                set_nested(evidence, path, value)
                self.rewrite_evidence_with_valid_hash(evidence)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    extract_run(self.run)

    def test_source_image_evidence_dimensions_mismatch_rejected_after_rehash(self):
        for dimension in ("height", "width"):
            with self.subTest(dimension=dimension):
                evidence = deepcopy(self.evidence)
                evidence["image"][dimension] += 1
                self.rewrite_evidence_with_valid_hash(evidence)
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    extract_run(self.run)

    def test_undecodable_source_image_rejected_even_with_matching_hash(self):
        path = self.run / self.entry["image_path"]
        path.write_bytes(b"not an image")
        manifest = deepcopy(self.manifest)
        manifest["frames"][0]["artifact_sha256"][self.entry["image_path"]] = digest(path)
        write_json(self.run / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            extract_run(self.run)

    def test_invalid_masks_in_real_npz_are_rejected_after_rehash(self):
        for bad in (np.ones((16, 12), np.uint8), np.ones((15, 12), bool)):
            with self.subTest(shape=bad.shape, dtype=str(bad.dtype)):
                path = self.run / self.entry["masks_path"]
                np.savez_compressed(path, **{"mask-left": bad, "mask-right": np.ones((16, 12), bool)})
                manifest = deepcopy(self.manifest)
                manifest["frames"][0]["artifact_sha256"][self.entry["masks_path"]] = digest(path)
                write_json(self.run / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "boolean.*dimensions"):
                    extract_run(self.run)

    def test_invalid_source_manifest_frame_sets_and_identities_are_rejected(self):
        mutations = [
            (("status",), "running"), (("frames",), []),
            (("frames", 1, "frame_id"), "late"), (("frames", 1, "frame_id"), "../escape"),
            (("frames", 1, "video_name"), "other.mp4"), (("frames", 1, "video_sha256"), "f" * 64),
            (("frames", 1, "frame_index"), 33), (("frames", 1, "capture_time_s"), 2.75),
            (("frames", 1, "capture_time_s"), -1), (("frames", 1, "capture_time_s"), True),
            (("frames", 1, "capture_time_s"), "4.5"),
        ]
        for path, value in mutations:
            with self.subTest(path=path, value=value):
                manifest = deepcopy(self.manifest)
                set_nested(manifest, path, value)
                write_json(self.run / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    extract_run(self.run)

    def test_payload_schema_or_source_manifest_hash_mismatch_is_rejected(self):
        for key, value in (("schema_version", "wrong-schema"), ("source_manifest_sha256", "0" * 64)):
            with self.subTest(key=key):
                changed = deepcopy(self.payload)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "does not match source run"):
                    self.load_changed(changed)

    def test_payload_frame_set_missing_extra_duplicate_or_renamed_is_rejected(self):
        frames = self.payload["frames"]
        renamed = deepcopy(frames)
        renamed[0]["frame_id"] = "alien"
        for name, value in (("missing", frames[:-1]), ("empty", []),
                            ("duplicate-extra", frames + [frames[0]]),
                            ("duplicate-replacing", [frames[0], frames[0], frames[2]]),
                            ("renamed", renamed)):
            with self.subTest(case=name):
                changed = deepcopy(self.payload)
                changed["frames"] = deepcopy(value)
                with self.assertRaisesRegex(ValueError, "frame set"):
                    self.load_changed(changed)

    def test_payload_source_identity_binding_and_dimensions_mismatches_are_rejected(self):
        mutations = [
            (("capture_time_s",), 99), (("source", "video_name"), "other.mp4"),
            (("source", "video_sha256"), "0" * 64), (("source", "frame_index"), 999),
            (("source", "artifact_sha256"), {}), (("image", "height"), 17), (("image", "width"), 13),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                changed = deepcopy(self.payload)
                set_nested(changed["frames"][0], path, value)
                with self.assertRaises(ValueError):
                    self.load_changed(changed)

    def test_boundary_evidence_frame_identity_mismatch_is_rejected(self):
        changed = deepcopy(self.payload)
        changed["frames"][0]["boundaries"][0]["evidence_frame_id"] = "late"
        with self.assertRaises(ValueError):
            self.load_changed(changed)

    def test_loader_replays_geometry_instead_of_trusting_source_hashes(self):
        for points in (
                [[1.5, -0.5], [1.5, 15.5]],
                [[5.0, -0.5], [5.0, 15.5]],
                [[1.5, 1.5], [9.5, 9.5]],
                [[5.5, -0.5], [5.5, 8.5]],
        ):
            with self.subTest(points=points):
                changed = deepcopy(self.payload)
                boundary = changed["frames"][0]["boundaries"][0]
                boundary["points_px"] = points
                boundary["length_px"] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                with self.assertRaisesRegex(ValueError, "source-mask replay"):
                    self.load_changed(changed)

    def test_loader_replays_candidate_types_endpoints_diagnostics_and_counts(self):
        mutations = [
            (("frames", 0, "boundaries", 0, "type"), "water_visible_interface"),
            (("frames", 0, "boundaries", 0, "endpoint_reasons"), [["unknown"], ["unknown"]]),
            (("frames", 0, "diagnostics", 0, "reason"), "occlusion"),
            (("frames", 0, "stats", "boundary_count"), 999),
            (("summary", "boundary_count"), 999),
            (("parameters", "gap_bridging_px"), 5),
            (("parameters", "min_length_px"), 100),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                changed = deepcopy(self.payload)
                set_nested(changed, path, value)
                with self.assertRaisesRegex(ValueError, "source-mask replay"):
                    self.load_changed(changed)

    def test_loader_rejects_unsupported_or_omitted_local_support(self):
        extra, _ = make_evidence((16, 12), [
            ("remote", "crop", rectangle((16, 12), 0, 4, 0, 2)),
            ("local", "crop", rectangle((16, 12), 4, 8, 2, 4)),
        ])
        self.evidence["instances"].extend(extra["instances"])
        self.rewrite_evidence_with_valid_hash(self.evidence)
        masks_path = self.run / self.entry["masks_path"]
        with np.load(masks_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays.update({"mask-remote": rectangle((16, 12), 0, 4, 0, 2),
                       "mask-local": rectangle((16, 12), 4, 8, 2, 4)})
        np.savez_compressed(masks_path, **arrays)
        manifest = json.loads((self.run / "manifest.json").read_bytes())
        manifest["frames"][0]["artifact_sha256"][self.entry["masks_path"]] = digest(masks_path)
        write_json(self.run / "manifest.json", manifest)
        payload = extract_run(self.run)
        self.assertEqual(self.load_changed(payload), payload)
        boundary = payload["frames"][1]["boundaries"][0]
        self.assertEqual(boundary["side_a"]["instance_ids"], ["left", "local"])
        for ids in (["left"], ["left", "local", "remote"]):
            with self.subTest(ids=ids):
                changed = deepcopy(payload)
                changed["frames"][1]["boundaries"][0]["side_a"]["instance_ids"] = ids
                with self.assertRaisesRegex(ValueError, "source-mask replay"):
                    self.load_changed(changed)

    def test_loader_replays_one_sided_contours_with_unknown_other_side(self):
        self.evidence["instances"] = self.evidence["instances"][:1]
        self.rewrite_evidence_with_valid_hash(self.evidence)
        payload = extract_run(self.run)
        self.assertEqual(self.load_changed(payload), payload)
        contour = payload["frames"][1]["coverage_contours"][0]
        self.assertEqual(contour["side_b"]["semantic"], "unknown")
        for x in (1.5, 8.5):
            changed = deepcopy(payload)
            changed["frames"][1]["coverage_contours"][0]["points_px"] = [[x, -0.5], [x, 15.5]]
            with self.subTest(x=x), self.assertRaisesRegex(ValueError, "source-mask replay"):
                self.load_changed(changed)

    def test_loader_rejects_invalid_threshold_and_missing_candidates(self):
        for value in (None, True, "12", -1):
            changed = deepcopy(self.payload)
            changed["parameters"]["min_length_px"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load_changed(changed)
        changed = deepcopy(self.payload)
        changed["frames"][0]["boundaries"] = []
        with self.assertRaisesRegex(ValueError, "source-mask replay"):
            self.load_changed(changed)

    def test_payload_frame_and_boundary_flags_must_all_be_literal_false(self):
        for scope in ((), ("frames", 0), ("frames", 0, "boundaries", 0)):
            for flag in FLAGS:
                for value in (True, 0, None, "false", "missing"):
                    with self.subTest(scope=scope, flag=flag, value=value):
                        changed = deepcopy(self.payload)
                        target = changed
                        for key in scope:
                            target = target[key]
                        if value == "missing":
                            del target[flag]
                        else:
                            target[flag] = value
                        with self.assertRaises(ValueError):
                            self.load_changed(changed)

    def test_duplicate_boundary_ids_and_unknown_support_ids_are_rejected(self):
        changed = deepcopy(self.payload)
        changed["frames"][0]["boundaries"].append(deepcopy(changed["frames"][0]["boundaries"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate boundary identity"):
            self.load_changed(changed)
        for side in ("side_a", "side_b"):
            changed = deepcopy(self.payload)
            changed["frames"][0]["boundaries"][0][side]["instance_ids"] = ["not-in-source"]
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, "Invalid boundary support"):
                self.load_changed(changed)

    def test_source_artifact_path_escape_is_rejected_before_hash_or_binding_checks(self):
        for field in ("image_path", "evidence_path", "masks_path"):
            outside = self.base / ("outside-" + Path(self.entry[field]).name)
            outside.write_bytes((self.run / self.entry[field]).read_bytes())
            for relative in ("../" + outside.name, outside.as_posix(),
                             "..\\" + outside.name, "C:/outside/file", "./" + self.entry[field]):
                with self.subTest(field=field, path=relative):
                    manifest = deepcopy(self.manifest)
                    manifest["frames"][0][field] = relative
                    manifest["frames"][0]["artifact_sha256"][relative] = digest(outside)
                    write_json(self.run / "manifest.json", manifest)
                    changed = deepcopy(self.payload)
                    changed["source_manifest_sha256"] = digest(self.run / "manifest.json")
                    with self.assertRaisesRegex(ValueError, "artifact path|escapes source run"):
                        extract_run(self.run)
                    with self.assertRaisesRegex(ValueError, "artifact path|escapes source run"):
                        self.load_changed(changed)

    def source_snapshot(self):
        return {path.relative_to(self.run).as_posix(): path.read_bytes()
                for path in self.run.rglob("*") if path.is_file()}

    def cli(self, output):
        return subprocess.run(
            [sys.executable, "-B", "-m", "field_boundary_mapping.evidence.extract_visible_boundaries",
             "--run", str(self.run), "--output", str(output)],
            cwd=str(ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def test_cli_writes_new_external_payload_without_mutating_source(self):
        before = self.source_snapshot()
        output = self.base / "cli-result.json"
        result = self.cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_boundary_payload(output, self.run), self.payload)
        self.assertEqual(json.loads(result.stdout.splitlines()[0]), self.payload["summary"])
        self.assertEqual(self.source_snapshot(), before)
        original = output.read_bytes()
        second = self.cli(output)
        self.assertEqual(second.returncode, 2, second.stderr)
        self.assertIn("new file outside the source run", second.stderr)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(self.source_snapshot(), before)

    def test_cli_refuses_existing_external_file_and_all_source_run_destinations(self):
        before = self.source_snapshot()
        sentinel = self.base / "existing.json"
        sentinel.write_bytes(b"do not overwrite")
        for output in (sentinel, self.run, self.run / "manifest.json",
                       self.run / self.entry["image_path"], self.run / "new-derived.json",
                       self.run / self.entry["frame_id"] / "new-derived.json"):
            with self.subTest(output=output):
                result = self.cli(output)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("new file outside the source run", result.stderr)
                self.assertEqual(sentinel.read_bytes(), b"do not overwrite")
                self.assertEqual(self.source_snapshot(), before)


class GapClosureRunTests(ContractAssertions):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="visible-boundaries-gap-run-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.run, self.manifest = write_synthetic_run(self.base)
        # Rewrite the "late" frame so crop and ground are separated by a 2px band.
        self.entry = self.manifest["frames"][0]
        masks_path = self.run / self.entry["masks_path"]
        with np.load(masks_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["mask-left"] = rectangle((16, 12), 0, 16, 0, 5)
        arrays["mask-right"] = rectangle((16, 12), 0, 16, 7, 12)
        np.savez_compressed(masks_path, **arrays)
        self.manifest = deepcopy(self.manifest)
        self.manifest["frames"][0]["artifact_sha256"][self.entry["masks_path"]] = digest(masks_path)
        write_json(self.run / "manifest.json", self.manifest)
        self.payload = extract_run(self.run, gap_closure_px=1)
        self.output = self.base / "boundaries.json"
        write_json(self.output, self.payload)
        self.frame_order = {f["frame_id"]: i for i, f in enumerate(self.payload["frames"])}

    def load_changed(self, payload):
        write_json(self.output, payload)
        return load_boundary_payload(self.output, self.run)

    def test_gap_closed_run_round_trip_and_summary(self):
        self.assertEqual(self.payload["parameters"]["gap_closure_px"], 1)
        self.assertEqual(self.payload["summary"]["gap_closed_count"], 1)
        self.assertEqual(self.payload["summary"]["frames_with_gap_closed"], 1)
        self.assertEqual(self.payload["summary"]["gap_closed_length_px"], 16.0)
        by_id = {f["frame_id"]: f for f in self.payload["frames"]}
        self.assertEqual(by_id["late"]["boundaries"], [])
        self.assertEqual(len(by_id["late"]["gap_closed_boundaries"]), 1)
        record = by_id["late"]["gap_closed_boundaries"][0]
        self.assertEqual(record["boundary_id"], "GC001")
        self.assertEqual(record["status"], "gap_closed_candidate")
        self.assertEqual(record["type"], "crop_ground_interface")
        self.assertEqual(record["side_a"], {"semantic": "crop", "instance_ids": ["left"], "source_semantics": ["crop"]})
        self.assertEqual(record["side_b"], {"semantic": "ground", "instance_ids": ["right"], "source_semantics": ["ground"]})
        self.assert_line(record, [[5.5, -0.5], [5.5, 15.5]])
        # Observed contact and empty frames stay untouched by the closure layer.
        self.assertEqual(by_id["early"]["boundaries"][0]["boundary_id"], "B001")
        self.assertEqual(by_id["early"]["gap_closed_boundaries"], [])
        self.assertEqual(by_id["empty"]["gap_closed_boundaries"], [])
        self.assertEqual(self.load_changed(self.payload), self.payload)

    def test_loader_rejects_gap_closure_tampering(self):
        index = self.frame_order["late"]
        mutations = [
            (("parameters", "gap_closure_px"), 0),
            (("frames", index, "gap_closed_boundaries", 0, "length_px"), 99.0),
            (("frames", index, "gap_closed_boundaries", 0, "type"), "vegetation_coverage_interface"),
            (("frames", index, "gap_closed_boundaries", 0, "status"), "image_candidate"),
            (("frames", index, "gap_closed_boundaries", 0, "points_px"), [[1.5, -0.5], [1.5, 15.5]]),
            (("frames", index, "gap_closed_boundaries", 0, "gap", "assigned_neighbor_px"), 1),
            (("frames", index, "gap_closed_boundaries", 0, "side_b", "instance_ids"), ["not-in-source"]),
            (("frames", index, "gap_closed_boundaries", 0, "semantic_verified"), True),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                changed = deepcopy(self.payload)
                set_nested(changed, path, value)
                with self.assertRaises(ValueError):
                    self.load_changed(changed)

    def test_cli_writes_gap_closed_payload(self):
        output = self.base / "cli-gap.json"
        result = subprocess.run(
            [sys.executable, "-B", "-m", "field_boundary_mapping.evidence.extract_visible_boundaries",
             "--run", str(self.run), "--output", str(output), "--gap-closure-px", "1"],
            cwd=str(ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_boundary_payload(output, self.run), self.payload)


if __name__ == "__main__":
    unittest.main()
