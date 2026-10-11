import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_boundary_mapping.scene.refine_scene_regions as refinement
from field_boundary_mapping.proposals.run_offline import file_hash, write_json
from field_boundary_mapping.proposals.run_proposals import REGION_PROMPTS
from field_boundary_mapping.scene.run_scene_understanding import instruction_for, load_scene_run
from field_boundary_mapping.scene.scene_understanding import GATES, INSTRUCTIONS, PROMPT_VERSION, ROLES, SCHEMA_VERSION, pixel_points, scene_queries
from field_boundary_mapping.semantic_pipeline import load_core


def region(region_id="r1", role="current_field", surface="ground", **changes):
    result = {"region_id": region_id, "role": role, "surface": surface,
              "polygon": [[100, 100], [800, 100], [800, 800], [100, 800]],
              "evidence": "Synthetic geometry, not real perception", "uncertainties": ["Function unverified"]}
    result.update(changes)
    return result


def scene(*regions, boundaries=None):
    return {"regions": list(regions), "relations": [], "boundaries": boundaries or [],
            "unknowns": ["Synthetic scene, physical extent unknown"]}


def boxed(region_id="r1", role="current_field", surface="ground", box=(100, 100, 800, 800), **changes):
    result = region(region_id, role, surface, **changes)
    del result["polygon"]
    result["box"] = list(box)
    return result


def detection(query, mask, **changes):
    result = {key: query[key] for key in ("proposal_id", "semantic", "prompt", "box_xyxy")}
    result.update(mask=mask, score=0.9, bbox_xyxy=[0, 0, mask.shape[1], mask.shape[0]])
    result.update(changes)
    return result


def gates_are_false(test, value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in GATES:
                test.assertIs(item, False, key)
            test.assertNotIn(key, ("world_map", "control"))
            gates_are_false(test, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            gates_are_false(test, item)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()  # Real C++ spatial checks; no model or GPU is used.

    def setUp(self):
        self.image = np.zeros((40, 40, 3), np.uint8)
        self.mask = np.zeros((40, 40), bool)
        self.mask[8:28, 8:28] = True

    def refine(self, value, masks=()):
        queries = scene_queries(value, 40, 40)
        detections = [detection(queries[0], mask) for mask in masks]
        return refinement.refine_regions(self.image, value, detections, self.core, "f0")

    def test_empty_mixed_and_unknown_preserve_coarse_without_fake_masks(self):
        for value in (scene(), scene(region(surface="mixed"), region("r2", "farm_road", "unknown"))):
            with self.subTest(value=value):
                evidence, raw, functional = self.refine(value)
                self.assertEqual(raw, {})
                self.assertEqual(functional, {})
                self.assertEqual(evidence["raw_evidence"]["queries"], [])
                for entry, original in zip(evidence["regions"], value["regions"]):
                    self.assertEqual(entry["status"], "unrefined_surface")
                    self.assertEqual(entry["polygon"], original["polygon"])
                    self.assertIsNone(entry["candidate_functional_mask_key"])
                    self.assertFalse(entry["segmentation_fallback_used"])
                views = refinement.render_overlays(self.image, evidence, functional)
                np.testing.assert_array_equal(views["refined_overlay.jpg"], self.image)
                gates_are_false(self, evidence)

    def test_roles_never_become_surface_prompts_or_ego_relations(self):
        value = scene(*(region(f"r{i}", role, "ground") for i, role in enumerate(ROLES)),
                      region("r6", "obstacle", "vehicle"), region("r7", "neighbor_field", "crop"))
        queries = scene_queries(value, 40, 40)
        self.assertEqual(len(queries), 8)
        for query, original in zip(queries, value["regions"]):
            self.assertEqual(query["proposal_id"], original["region_id"])
            self.assertEqual(query["semantic"], original["surface"])
            self.assertEqual(query["prompt"], REGION_PROMPTS[original["surface"]])
            self.assertEqual(query["relation"], "unspecified")
        detections = [detection(query, self.mask) for query in queries]
        evidence, raw, functional = refinement.refine_regions(self.image, value, detections, self.core, "f0")
        self.assertEqual(len(raw), 8)
        self.assertEqual(len(functional), 8)
        for entry, original in zip(evidence["regions"], value["regions"]):
            self.assertEqual((entry["role"], entry["surface"]), (original["role"], original["surface"]))
        vehicle = evidence["raw_evidence"]["instances"][6]
        self.assertEqual(vehicle["semantic"], "vehicle")
        self.assertFalse(vehicle["suppressed_as_ego"])
        self.assertFalse(vehicle["ego_vehicle_verified"])
        gates_are_false(self, evidence)
        value["regions"].append(region("r8", surface="water"))
        with self.assertRaises(ValueError):
            self.refine(value)

    def test_union_intersection_keeps_every_raw_mask_without_mutating_inputs(self):
        value = scene(region(polygon=[[100, 100], [800, 100], [100, 800]]))
        left, right = np.zeros((40, 40), bool), np.zeros((40, 40), bool)
        left[4:32, 4:18] = True
        left[2:4, 4:8] = True  # Allowed box leakage, retained in raw but clipped in candidate.
        right[4:32, 18:32] = True
        rejected = np.ones((40, 40), bool)
        masks = [left, right] + [self.mask.copy() for _ in range(10)] + [rejected]
        originals = [mask.copy() for mask in masks]
        original_scene = copy.deepcopy(value)
        for mask in masks:
            mask.flags.writeable = False
        evidence, raw, functional = self.refine(value, masks)
        entry = evidence["regions"][0]
        self.assertEqual(len(raw), 13)
        self.assertEqual(len(evidence["raw_evidence"]["instances"]), 13)
        self.assertEqual(len(entry["raw_ids"]), 13)
        self.assertEqual(len(entry["pass_ids"]), 12)
        self.assertEqual(entry["status"], "refined_candidate")
        polygon = np.zeros((40, 40), np.uint8)
        cv2.fillPoly(polygon, [np.asarray(pixel_points(value["regions"][0]["polygon"], 40, 40), np.int32)], 1)
        union = left | right | self.mask
        expected = union & polygon.astype(bool)
        np.testing.assert_array_equal(functional[entry["candidate_functional_mask_key"]], expected)
        for actual, original, mask in zip(raw.values(), originals, masks):
            np.testing.assert_array_equal(actual, original)
            np.testing.assert_array_equal(mask, original)
        self.assertEqual(value, original_scene)
        self.assertFalse(self.image.any())
        self.assertAlmostEqual(entry["passed_union_overlap_fraction"], expected.sum() / union.sum())
        self.assertAlmostEqual(entry["coarse_polygon_support_fraction"], expected.sum() / polygon.sum())
        self.assertLess(entry["passed_union_overlap_fraction"], 1)
        self.assertIn("clipped by VLM", entry["extent_caveat"])
        self.assertIsNone(evidence["raw_evidence"]["instance_cap"])
        self.assertFalse(evidence["raw_evidence"]["mask_clipping_applied"])
        gates_are_false(self, evidence)
        json.dumps(evidence, allow_nan=False)

    def test_spatial_rejection_empty_mask_and_no_detection_remain_unrefined(self):
        for masks, status in (([], "no_detection"), ([np.ones((40, 40), bool)], "spatial_rejected"),
                              ([np.zeros((40, 40), bool)], "spatial_rejected")):
            with self.subTest(status=status, count=len(masks)):
                evidence, raw, functional = self.refine(scene(region()), masks)
                entry = evidence["regions"][0]
                self.assertEqual(entry["status"], status)
                self.assertEqual(entry["pass_ids"], [])
                self.assertEqual(len(raw), len(masks))
                self.assertEqual(functional, {})
                self.assertIsNone(entry["candidate_functional_mask_key"])
                self.assertEqual(entry["candidate_mask_contours"], [])
                self.assertTrue(entry["coarse_polygon_preserved"])

    def test_box_pass_with_no_polygon_overlap_does_not_invent_segmentation(self):
        value = scene(region(polygon=[[100, 100], [800, 100], [100, 800]]))
        mask = np.zeros((40, 40), bool)
        mask[27:30, 27:30] = True
        evidence, raw, functional = self.refine(value, [mask])
        entry = evidence["regions"][0]
        self.assertEqual(entry["pass_ids"], ["candidate_0000"])
        self.assertEqual(entry["status"], "spatial_rejected")
        self.assertEqual(entry["reason"], "box_consistent_masks_have_no_polygon_support")
        self.assertEqual(entry["passed_union_overlap_fraction"], 0)
        self.assertEqual(functional, {})
        np.testing.assert_array_equal(raw["candidate_0000"], mask)

    def test_box_localization_uses_the_same_coarse_envelope_as_a_polygon(self):
        value = scene(boxed())
        queries = scene_queries(value, 40, 40)
        self.assertEqual(queries[0]["box_xyxy"], [4, 4, 32, 32])
        evidence, raw, functional = self.refine(value, (self.mask,))
        entry = evidence["regions"][0]
        self.assertEqual(entry["coarse_geometry"], "vlm_box")
        self.assertNotIn("polygon", entry)
        self.assertEqual(entry["coarse_polygon_px"], [[4, 4], [32, 4], [32, 32], [4, 32]])
        self.assertEqual(entry["status"], "refined_candidate")
        self.assertEqual(entry["candidate_pixels"], int(self.mask.sum()))
        np.testing.assert_array_equal(functional["candidate_functional_r1"], self.mask)
        polygon_evidence, _, polygon_functional = self.refine(scene(region()), (self.mask,))
        self.assertEqual(polygon_evidence["regions"][0]["coarse_geometry"], "vlm_polygon")
        self.assertEqual(polygon_evidence["regions"][0]["coarse_polygon_px"], entry["coarse_polygon_px"])
        self.assertEqual(polygon_evidence["regions"][0]["coarse_polygon_pixels"],
                         entry["coarse_polygon_pixels"])
        np.testing.assert_array_equal(polygon_functional["candidate_functional_r1"],
                                      functional["candidate_functional_r1"])
        gates_are_false(self, evidence)

    def test_detection_binding_and_invalid_masks_scores_are_rejected(self):
        value = scene(region(), region("mixed", "farm_road", "mixed"))
        query = scene_queries(value, 40, 40)[0]
        invalid = [{"proposal_id": "other"}, {"proposal_id": "mixed"}, {"semantic": "current_field"},
                   {"semantic": "crop"}, {"prompt": "headland"}, {"box_xyxy": [0, 0, 40, 40]},
                   {"mask": np.ones((3, 3), bool)}, {"mask": self.mask.astype(np.uint8)},
                   {"score": True}, {"score": float("nan")}, {"score": -0.1}, {"score": 1.1}]
        for change in invalid:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                bound = {**detection(query, self.mask), **change}
                refinement.refine_regions(self.image, value, [bound], self.core, "f0")
        with self.assertRaises(ValueError):
            self.refine(scene(region(role="ground", surface="current_field")))

    def test_overlaps_are_counted_and_not_erased_or_resolved_by_region_order(self):
        value = scene(region("field"), region("road", "farm_road"), region("end", "headland_candidate"))
        queries = scene_queries(value, 40, 40)
        detections = [detection(query, self.mask) for query in queries]
        evidence, _, functional = refinement.refine_regions(self.image, value, detections, self.core, "f0")
        overlap = evidence["overlaps"]["candidate"]
        self.assertTrue(overlap["has_overlap"])
        self.assertEqual(overlap["overlap_pixels"], 400)  # Triple overlap counted once globally.
        self.assertEqual(len(overlap["pairs"]), 3)
        self.assertEqual(len(functional), 3)
        for entry in evidence["regions"]:
            self.assertEqual(len(entry["candidate_overlapping_region_ids"]), 2)
            np.testing.assert_array_equal(functional[entry["candidate_functional_mask_key"]], self.mask)
        views = refinement.render_overlays(self.image, evidence, functional)
        value["regions"].reverse()
        reordered, _, other_masks = refinement.refine_regions(self.image, value, detections, self.core, "f0")
        self.assertEqual(reordered["overlaps"], evidence["overlaps"])
        other_views = refinement.render_overlays(self.image, reordered, other_masks)
        for name in views:
            np.testing.assert_array_equal(views[name], other_views[name])
        overlap_blend = (0.5 * self.image[20, 20] + 0.5 * np.mean(
            [refinement.ROLE_PALETTE[role][::-1] for role in ("current_field", "farm_road", "headland_candidate")],
            axis=0)).astype(np.uint8)
        np.testing.assert_array_equal(views["refined_overlay.jpg"][20, 20], overlap_blend)
        self.assertEqual(set(refinement.ROLE_PALETTE), set(ROLES))
        self.assertEqual(len(set(refinement.ROLE_PALETTE.values())), len(ROLES))

    def test_vlm_relations_and_open_boundaries_stay_separate_from_mask_contours(self):
        boundary = {"boundary_id": "b1", "side_a": "r1", "side_b": "r2", "kind": "field_headland",
                    "points": [[200, 125], [675, 125]], "evidence": "Synthetic visible segment"}
        value = scene(region(), region("r2", "headland_candidate", "mixed"), boundaries=[boundary])
        value["relations"] = [{"subject": "r2", "relation": "headland_of", "object": "r1", "evidence": "Hypothesis"}]
        evidence, _, functional = self.refine(value, [self.mask])
        self.assertEqual(evidence["vlm_relations"], value["relations"])
        self.assertEqual(evidence["vlm_boundaries"], [boundary])
        diagnostic, = evidence["boundary_geometry_diagnostics"]
        self.assertEqual(diagnostic["mean_distance_px"], 3)
        self.assertEqual(diagnostic["maximum_distance_px"], 3)
        self.assertEqual(diagnostic["status"], "geometry_diagnostic_only")
        self.assertFalse(diagnostic["boundary_refinement_performed"])
        with patch.object(cv2, "putText", side_effect=AssertionError("No image text")), \
                patch.object(cv2, "rectangle", side_effect=AssertionError("No boxes")), \
                patch.object(cv2, "polylines", wraps=cv2.polylines) as lines:
            refinement.render_overlays(self.image, evidence, functional)
            self.assertTrue(lines.call_args_list)
            self.assertTrue(all(call.args[2] is False for call in lines.call_args_list))
        boundary["points"] = [[200, 200], [700, 200], [700, 700]]
        evidence, _, functional = self.refine(value)
        view = refinement.render_overlays(self.image, evidence, functional)["boundaries_overlay.jpg"]
        np.testing.assert_array_equal(view[18, 18], self.image[18, 18])  # No invented closing diagonal.

    def test_ambiguous_boundary_geometry_is_omitted_not_refined(self):
        boundary = {"boundary_id": "b1", "side_a": "r1", "side_b": "unknown", "kind": "unknown",
                    "points": [[200, 125], [675, 125]], "evidence": "Synthetic segment"}
        disconnected = self.mask.copy()
        disconnected[29:31, 29:31] = True
        hole = self.mask.copy()
        hole[12:20, 12:20] = False
        for mask in (disconnected, hole):
            evidence, _, _ = self.refine(scene(region(), boundaries=[boundary]), [mask])
            self.assertEqual(evidence["boundary_geometry_diagnostics"], [])
        for points in ([[100, 100], [800, 800], [100, 800], [800, 100]], [[100, 100], [101, 101]]):
            boundary["points"] = points
            evidence, _, _ = self.refine(scene(region(), boundaries=[boundary]), [self.mask])
            self.assertEqual(evidence["boundary_geometry_diagnostics"], [])
            self.assertEqual(evidence["vlm_boundaries"][0]["points"], points)

    def test_diagonal_polygon_cut_uses_passed_union_not_individual_or_rejected_masks(self):
        boundaries = [
            {"boundary_id": "cut", "side_a": "r1", "side_b": "unknown", "kind": "unknown",
             "points": [[375, 525], [525, 375]], "evidence": "Synthetic diagonal polygon cut"},
            {"boundary_id": "sam", "side_a": "unknown", "side_b": "r1", "kind": "unknown",
             "points": [[250, 200], [625, 200]], "evidence": "Synthetic SAM edge"},
        ]
        value = scene(region(polygon=[[100, 100], [800, 100], [100, 800]]), boundaries=boundaries)
        original_scene = copy.deepcopy(value)
        left, right = self.mask.copy(), self.mask.copy()
        left[:, 18:] = False
        right[:, :18] = False
        masks = [left, right, np.ones_like(self.mask)]
        originals = [mask.copy() for mask in masks]
        for mask in masks:
            mask.flags.writeable = False
        evidence, raw, functional = self.refine(value, masks)
        entry = evidence["regions"][0]
        self.assertEqual(len(entry["pass_ids"]), 2)
        polygon = refinement._polygon_mask(pixel_points(value["regions"][0]["polygon"], 40, 40), (40, 40))
        candidate = self.mask & polygon
        np.testing.assert_array_equal(functional[entry["candidate_functional_mask_key"]], candidate)
        edges = [(y, x) for y in range(1, 39) for x in range(1, 39)
                 if candidate[y, x] and not candidate[y - 1:y + 2, x - 1:x + 2].all()]
        cuts = sum(bool(self.mask[y - 1:y + 2, x - 1:x + 2].all()) for y, x in edges)
        self.assertGreater(cuts, 0)
        self.assertLess(cuts, len(edges))
        self.assertEqual(entry["candidate_edge_pixels"], len(edges))
        self.assertEqual(entry["polygon_cut_edge_pixels"], cuts)
        self.assertEqual(entry["polygon_cut_edge_fraction"], cuts / len(edges))
        self.assertEqual(entry["image_border_edge_pixels"], 0)
        self.assertIn("polygon_clipping_edge_present", entry["quality_flags"])
        cut, sam = evidence["boundary_support_diagnostics"]
        self.assertEqual((cut["side"], sam["side"]), ("side_a", "side_b"))
        self.assertEqual(cut["sample_count"], 7)
        self.assertAlmostEqual(cut["mean_distance_px"], 51 / 7, places=6)
        self.assertEqual(cut["maximum_distance_px"], 9)
        self.assertEqual(cut["supported_fraction"], 0)
        self.assertEqual(sam["mean_distance_px"], 0)
        self.assertEqual(sam["maximum_distance_px"], 0)
        self.assertEqual(sam["supported_fraction"], 1)
        for diagnostic in (cut, sam):
            self.assertEqual(diagnostic["threshold_px"], 2)
            self.assertEqual(diagnostic["status"], "geometry_diagnostic_only")
            self.assertEqual(diagnostic["image_border_sample_fraction"], 0)
            self.assertIn("unclipped", diagnostic["edge_extent"])
        for actual, original, mask in zip(raw.values(), originals, masks):
            np.testing.assert_array_equal(actual, original)
            np.testing.assert_array_equal(mask, original)
        self.assertEqual(value, original_scene)
        self.assertEqual(evidence["vlm_boundaries"], boundaries)
        gates_are_false(self, evidence)
        json.dumps(evidence, allow_nan=False)

    def test_true_sam_edges_are_not_polygon_cut_and_tiny_regions_are_retained(self):
        evidence, _, _ = self.refine(scene(region()), [self.mask])
        entry = evidence["regions"][0]
        self.assertEqual(entry["candidate_edge_pixels"], 76)
        self.assertEqual(entry["polygon_cut_edge_pixels"], 0)
        self.assertEqual(entry["polygon_cut_edge_fraction"], 0)
        self.assertEqual(entry["quality_flags"], [])
        value = scene(region(), region("tiny", "farm_road"))
        tiny = np.zeros_like(self.mask)
        tiny[10, 10] = True
        queries = scene_queries(value, 40, 40)
        evidence, _, functional = refinement.refine_regions(
            self.image, value, [detection(queries[0], self.mask), detection(queries[1], tiny)], self.core, "f0")
        self.assertEqual(len(functional), 2)
        self.assertEqual(evidence["regions"][1]["candidate_pixels"], 1)
        for entry in evidence["regions"]:
            self.assertEqual(entry["quality_flags"], ["functional_overlap_unresolved"])
            self.assertEqual(entry["status"], "refined_candidate")
        self.assertIn("does not establish true or false", evidence["quality_flag_policy"])
        gates_are_false(self, evidence)

    def test_image_frame_edges_are_counted_separately_and_not_distance_targets(self):
        boundary = {"boundary_id": "frame", "side_a": "r1", "side_b": "unknown", "kind": "unknown",
                    "points": [[0, 250], [100, 250]], "evidence": "Synthetic frame contact"}
        value = scene(region(polygon=[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]), boundaries=[boundary])
        mask = np.zeros_like(self.mask)
        mask[:, :20] = True
        evidence, _, _ = self.refine(value, [mask])
        entry = evidence["regions"][0]
        self.assertEqual(entry["candidate_edge_pixels"], 38)
        self.assertEqual(entry["image_border_edge_pixels"], 78)
        self.assertEqual(entry["polygon_cut_edge_pixels"], 0)
        diagnostic, = evidence["boundary_support_diagnostics"]
        self.assertEqual(diagnostic["sample_count"], 5)
        self.assertEqual(diagnostic["image_border_sample_fraction"], 1 / 5)
        self.assertEqual(diagnostic["mean_distance_px"], 17)
        self.assertEqual(diagnostic["maximum_distance_px"], 19)
        self.assertEqual(diagnostic["supported_fraction"], 0)
        boundary["points"] = [[0, 0], [1000, 0]]
        evidence, _, _ = self.refine(value, [np.ones_like(self.mask)])
        entry = evidence["regions"][0]
        self.assertEqual(entry["candidate_edge_pixels"], 0)
        self.assertEqual(entry["image_border_edge_pixels"], 156)
        self.assertEqual(entry["polygon_cut_edge_pixels"], 0)
        self.assertIsNone(entry["polygon_cut_edge_fraction"])
        diagnostic, = evidence["boundary_support_diagnostics"]
        self.assertEqual(diagnostic["status"], "no_interior_edges")
        self.assertEqual(diagnostic["image_border_sample_fraction"], 1)
        for key in ("mean_distance_px", "maximum_distance_px", "supported_fraction"):
            self.assertIsNone(diagnostic[key])
        json.dumps(evidence, allow_nan=False)

    def test_empty_unions_and_degenerate_lines_record_every_named_side(self):
        boundary = {"boundary_id": "b1", "side_a": "r1", "side_b": "r2", "kind": "unknown",
                    "points": [[200, 125], [675, 125]], "evidence": "Synthetic segment"}
        value = scene(region(), region("r2", surface="mixed"), boundaries=[boundary])
        for masks in ([], [np.zeros_like(self.mask)], [np.ones_like(self.mask)]):
            with self.subTest(mask_count=len(masks)):
                evidence, raw, functional = self.refine(value, masks)
                self.assertEqual(len(raw), len(masks))
                self.assertEqual(functional, {})
                self.assertEqual(len(evidence["boundary_support_diagnostics"]), 2)
                for entry in evidence["regions"]:
                    self.assertEqual(entry["candidate_edge_pixels"], 0)
                    self.assertEqual(entry["polygon_cut_edge_pixels"], 0)
                    self.assertEqual(entry["image_border_edge_pixels"], 0)
                    self.assertIsNone(entry["polygon_cut_edge_fraction"])
                for diagnostic in evidence["boundary_support_diagnostics"]:
                    self.assertEqual(diagnostic["status"], "empty_passed_union")
                    self.assertGreater(diagnostic["sample_count"], 0)
                    for key in ("mean_distance_px", "maximum_distance_px", "supported_fraction"):
                        self.assertIsNone(diagnostic[key])
                gates_are_false(self, evidence)
                json.dumps(evidence, allow_nan=False)
        boundary["points"] = [[100, 100], [101, 101]]  # Distinct VLM points collapse to one pixel.
        evidence, _, _ = self.refine(value, [self.mask])
        self.assertEqual([(d["side"], d["region_id"]) for d in evidence["boundary_support_diagnostics"]],
                         [("side_a", "r1"), ("side_b", "r2")])
        for diagnostic in evidence["boundary_support_diagnostics"]:
            self.assertEqual(diagnostic["status"], "degenerate_polyline")
            self.assertEqual(diagnostic["sample_count"], 0)
            for key in ("mean_distance_px", "maximum_distance_px", "supported_fraction", "image_border_sample_fraction"):
                self.assertIsNone(diagnostic[key])
        self.assertEqual(evidence["vlm_boundaries"], [boundary])

    def test_unclipped_support_survives_zero_candidate_polygon_support(self):
        mask = np.zeros_like(self.mask)
        mask[27:30, 27:30] = True
        boundary = {"boundary_id": "b1", "side_a": "r1", "side_b": "unknown", "kind": "unknown",
                    "points": [[675, 675], [725, 675]], "evidence": "SAM outside triangular polygon"}
        value = scene(region(polygon=[[100, 100], [800, 100], [100, 800]]), boundaries=[boundary])
        evidence, raw, functional = self.refine(value, [mask])
        self.assertEqual(functional, {})
        self.assertEqual(evidence["regions"][0]["passed_union_pixels"], 9)
        self.assertEqual(evidence["regions"][0]["candidate_pixels"], 0)
        diagnostic, = evidence["boundary_support_diagnostics"]
        self.assertEqual(diagnostic["status"], "geometry_diagnostic_only")
        self.assertEqual(diagnostic["maximum_distance_px"], 0)
        self.assertEqual(diagnostic["supported_fraction"], 1)
        np.testing.assert_array_equal(raw["candidate_0000"], mask)

    def test_distance_support_includes_concavities_components_and_holes(self):
        disconnected = self.mask.copy()
        disconnected[29:31, 29:31] = True
        hole = self.mask.copy()
        hole[12:20, 12:20] = False
        concave = self.mask.copy()
        concave[16:28, 16:28] = False
        cases = ((disconnected, [[725, 725], [725, 750]]),
                 (hole, [[275, 300], [275, 475]]), (concave, [[375, 450], [375, 625]]))
        for mask, points in cases:
            with self.subTest(points=points):
                boundary = {"boundary_id": "b1", "side_a": "r1", "side_b": "unknown", "kind": "unknown",
                            "points": points, "evidence": "Synthetic nonconvex geometry"}
                evidence, _, functional = self.refine(scene(region(), boundaries=[boundary]), [mask])
                self.assertEqual(evidence["boundary_geometry_diagnostics"], [])  # Legacy omission unchanged.
                diagnostic, = evidence["boundary_support_diagnostics"]
                self.assertEqual(diagnostic["status"], "geometry_diagnostic_only")
                self.assertEqual(diagnostic["maximum_distance_px"], 0)
                self.assertEqual(diagnostic["supported_fraction"], 1)
                self.assertEqual(evidence["regions"][0]["polygon_cut_edge_pixels"], 0)
                np.testing.assert_array_equal(functional["candidate_functional_r1"], mask)
                gates_are_false(self, evidence)


class RasterSupportTests(unittest.TestCase):
    def diagnostic(self, mask, points):
        boundary = {"boundary_id": "b1", "side_a": "r1", "side_b": "unknown", "points": points}
        result, = refinement._boundary_support_diagnostics([boundary], {"r1": mask}, mask.shape[1], mask.shape[0])
        return result

    def test_exact_euclidean_distance_and_scaled_inclusive_threshold(self):
        mask = np.zeros((40, 40), bool)
        mask[10, 10] = True
        diagnostic = self.diagnostic(mask, [[300, 275], [325, 275]])  # Pixels (12,11), (13,11).
        self.assertAlmostEqual(diagnostic["mean_distance_px"], (np.sqrt(5) + np.sqrt(10)) / 2, places=6)
        self.assertAlmostEqual(diagnostic["maximum_distance_px"], np.sqrt(10), places=6)
        self.assertEqual(diagnostic["supported_fraction"], 0)
        diagnostic = self.diagnostic(mask, [[250, 200], [250, 150]])  # Distances 2, 3, 4 pixels.
        self.assertEqual(diagnostic["mean_distance_px"], 3)
        self.assertEqual(diagnostic["supported_fraction"], 1 / 3)
        mask = np.zeros((600, 800), bool)
        mask[100:200, 100:200] = True
        diagnostic = self.diagnostic(mask, [[125, 158], [200, 158]])  # y=95, five pixels above SAM.
        self.assertEqual(diagnostic["threshold_px"], 5)
        self.assertEqual(diagnostic["mean_distance_px"], 5)
        self.assertEqual(diagnostic["supported_fraction"], 1)

    def test_open_raster_is_not_closed_or_silently_omitted_for_crossings(self):
        mask = np.zeros((40, 40), bool)
        mask[8:28, 8:28] = True
        diagnostic = self.diagnostic(mask, [[200, 200], [675, 200], [675, 675]])
        self.assertEqual(diagnostic["sample_count"], 39)  # Shared vertex counted once; no closing diagonal.
        self.assertEqual(diagnostic["maximum_distance_px"], 0)
        self.assertEqual(diagnostic["supported_fraction"], 1)
        diagnostic = self.diagnostic(mask, [[200, 200], [675, 675], [200, 675], [675, 200]])
        self.assertEqual(diagnostic["status"], "geometry_diagnostic_only")
        self.assertGreater(diagnostic["maximum_distance_px"], 2)
        self.assertLess(diagnostic["supported_fraction"], 1)

    def test_single_pixel_wide_images_have_only_frame_edges(self):
        for shape in ((1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=shape):
                mask = np.ones(shape, bool)
                diagnostics = refinement._candidate_edge_diagnostics(mask, mask)
                self.assertEqual(diagnostics["candidate_edge_pixels"], 0)
                self.assertEqual(diagnostics["image_border_edge_pixels"], mask.size)
                self.assertEqual(diagnostics["polygon_cut_edge_pixels"], 0)
                self.assertIsNone(diagnostics["polygon_cut_edge_fraction"])


class SceneRefinementRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scene-refinement-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.scenes = self.root / "scenes"
        self.source.mkdir()
        self.scenes.mkdir()
        self.image = np.zeros((40, 40, 3), np.uint8)
        self.image_bytes = cv2.imencode(".png", self.image)[1].tobytes()
        self.weights = self.root / "synthetic.pt"
        self.weights.write_bytes(b"SYNTHETIC TEST ONLY, NEVER LOADED")
        self.args = argparse.Namespace(results=self.source, scenes=self.scenes, output=self.root / "out",
                                       weights=self.weights, clip_dir=None, confidence=0.3, device="cuda", threads=4)
        self.archive([scene(region())])

    def archive(self, values):
        frames, records = [], []
        for index, value in enumerate(values):
            frame_id = f"f{index}"
            image_name = frame_id + ".png"
            (self.source / image_name).write_bytes(self.image_bytes)
            frame = {"frame_id": frame_id, "video_name": "synthetic.mp4", "video_sha256": "b" * 64,
                     "frame_index": index, "capture_time_s": index + 0.5, "image_path": image_name,
                     "artifact_sha256": {image_name: hashlib.sha256(self.image_bytes).hexdigest()}}
            frames.append(frame)
            context = [{key: frame[key] for key in ("frame_id", "video_sha256", "frame_index", "capture_time_s")}]
            context[0].update(image_sha256=frame["artifact_sha256"][image_name], width=40, height=40, mime_type="image/png")
            record = {"schema_version": SCHEMA_VERSION, "frame_id": frame_id, "context": context,
                      "instruction": instruction_for(context), "status": "parsed", "raw_response": json.dumps(value),
                      "scene": value, "timing": {"finish_reason": "stop"}, **GATES}
            write_json(self.scenes / (frame_id + ".json"), record)
            records.append({"frame_id": frame_id, "sha256": file_hash(self.scenes / (frame_id + ".json"))})
        write_json(self.source / "manifest.json", {"schema_version": "field_boundary_run_v1", "frames": frames})
        self.scene_manifest = {
            "schema_version": "field_functional_scene_run_v1", "status": "completed", **GATES,
            "configuration": {"source_manifest_sha256": file_hash(self.source / "manifest.json"),
                              "prompt_version": PROMPT_VERSION,
                              "instruction_sha256": hashlib.sha256(INSTRUCTIONS[PROMPT_VERSION].encode()).hexdigest(),
                              "target_frames": [f["frame_id"] for f in frames], "mode": "single",
                              "history_frames": 2, "history_seconds": 2.0}, "frames": records,
        }
        write_json(self.scenes / "manifest.json", self.scene_manifest)

    def snapshot(self):
        return {str(p): p.read_bytes() for root in (self.source, self.scenes) for p in root.rglob("*") if p.is_file()}

    def test_runner_batches_once_per_queried_frame_and_writes_hashed_archives_progressively(self):
        values = [scene(region()), scene(region(surface="mixed")), scene(region(), region("r2", "farm_road"))]
        self.archive(values)
        before = self.snapshot()
        calls = []

        def predict(image, queries):
            self.assertEqual(os.environ.get("YOLO_OFFLINE"), "true")
            with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
                socket.getaddrinfo("offline-test.invalid", 443)
            partial = json.loads((self.args.output / "manifest.json").read_bytes())
            self.assertEqual(partial["status"], "running")
            self.assertEqual(len(partial["frames"]), 0 if not calls else 2)
            calls.append(queries)
            mask = np.zeros(image.shape[:2], bool)
            mask[8:28, 8:28] = True
            return ([detection(query, mask) for query in queries for _ in range(10)]
                    + [detection(queries[0], np.ones(image.shape[:2], bool))], {"total_seconds": 0.0})

        with patch.object(refinement, "LocalSAM3") as model:
            model.return_value.metadata = {"backend": "wrong", "prompt_version": "wrong", "prompts": ["wrong"], "vlm_used": True}
            model.return_value.predict_regions.side_effect = predict
            manifest = refinement.run(self.args)
            model.assert_called_once()
            self.assertEqual(model.return_value.predict_regions.call_count, 2)
        self.assertEqual([len(queries) for queries in calls], [1, 2])
        self.assertEqual(manifest["schema_version"], "field_functional_refinement_run_v1")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(before, self.snapshot())
        config = manifest["configuration"]
        self.assertEqual(config["weights_sha256"], file_hash(self.weights))
        self.assertEqual(config["source_scene_manifest_sha256"], file_hash(self.scenes / "manifest.json"))
        self.assertIn(Path(load_core().__file__).name, config["implementation_sha256"])
        self.assertIn("refine_scene_regions.py", config["implementation_sha256"])
        self.assertIn("run_proposals.py", config["implementation_sha256"])
        for index, entry in enumerate(manifest["frames"]):
            self.assertEqual(len(entry["artifact_sha256"]), 8)
            for relative, digest in entry["artifact_sha256"].items():
                self.assertEqual(file_hash(self.args.output / relative), digest)
            self.assertEqual((self.args.output / entry["image_path"]).read_bytes(), self.image_bytes)
            evidence = json.loads((self.args.output / entry["evidence_path"]).read_bytes())
            raw_evidence = json.loads((self.args.output / entry["raw_evidence_path"]).read_bytes())
            self.assertEqual(raw_evidence, evidence["raw_evidence"])
            self.assertEqual(evidence["source_scene_record_sha256"], file_hash(self.scenes / f"f{index}.json"))
            self.assertEqual(evidence["model"]["backend"], refinement.BACKEND)
            self.assertEqual(evidence["model"]["prompt_version"], refinement.PIPELINE_VERSION)
            self.assertEqual(evidence["model"]["prompts"], scene_queries(values[index], 40, 40))
            self.assertEqual(evidence["model"]["inference_performed"], index != 1)
            self.assertFalse(evidence["model"]["vlm_used"])
            self.assertEqual(evidence["display"]["default"], "refined_overlay.jpg")
            with np.load(self.args.output / entry["raw_masks_path"], allow_pickle=False) as raw, \
                    np.load(self.args.output / entry["functional_masks_path"], allow_pickle=False) as functional:
                self.assertEqual(len(raw.files), [11, 0, 21][index])
                self.assertEqual(len(functional.files), [1, 0, 2][index])
                self.assertEqual(len(raw_evidence["instances"]), len(raw.files))
                if raw.files:
                    self.assertTrue(raw[raw.files[-1]].all())  # Rejected raw mask was not clipped or dropped.
            gates_are_false(self, evidence)
        gates_are_false(self, manifest)

    def test_no_queries_never_loads_model_and_still_exports_frames(self):
        self.archive([scene(), scene(region(surface="mixed"), region("r2", "unknown", "unknown"))])
        with patch.object(refinement, "LocalSAM3") as model:
            manifest = refinement.run(self.args)
            model.assert_not_called()
        for entry in manifest["frames"]:
            evidence = json.loads((self.args.output / entry["evidence_path"]).read_bytes())
            self.assertFalse(evidence["model"]["inference_performed"])
            self.assertEqual(evidence["model"]["prompts"], [])
            self.assertEqual(evidence["timing"]["status"], "skipped_no_surface_queries")
            with np.load(self.args.output / entry["functional_masks_path"], allow_pickle=False) as masks:
                self.assertEqual(masks.files, [])

    def test_loader_bindings_fail_before_model_or_output(self):
        for mismatch in ("image", "source_manifest", "record_hash", "record_frame", "context", "scene", "gate", "incomplete"):
            with self.subTest(mismatch=mismatch):
                self.archive([scene(region())])
                record_path = self.scenes / "f0.json"
                record = json.loads(record_path.read_bytes())
                if mismatch == "image":
                    (self.source / "f0.png").write_bytes(b"changed original")
                elif mismatch == "source_manifest":
                    self.scene_manifest["configuration"]["source_manifest_sha256"] = "a" * 64
                elif mismatch == "record_hash":
                    self.scene_manifest["frames"][0]["sha256"] = "a" * 64
                elif mismatch == "incomplete":
                    self.scene_manifest["status"] = "running"
                else:
                    if mismatch == "record_frame":
                        record["frame_id"] = "other"
                    elif mismatch == "context":
                        record["context"][0]["image_sha256"] = "a" * 64
                    elif mismatch == "scene":
                        record["scene"]["regions"][0]["role"] = "farm_road"
                    elif mismatch == "gate":
                        record["semantic_verified"] = True
                    write_json(record_path, record)
                    self.scene_manifest["frames"][0]["sha256"] = file_hash(record_path)
                write_json(self.scenes / "manifest.json", self.scene_manifest)
                with patch.object(refinement, "LocalSAM3") as model, self.assertRaises(ValueError):
                    refinement.run(self.args)
                model.assert_not_called()
                self.assertFalse(self.args.output.exists())

    def test_existing_and_nested_outputs_are_refused_without_touching_inputs(self):
        before = self.snapshot()
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_bytes(b"untouched")
        for output in (existing, self.source, self.scenes, self.source / "new", self.scenes / "new"):
            self.args.output = output
            with self.subTest(output=output), patch.object(refinement, "LocalSAM3") as model, \
                    self.assertRaisesRegex(ValueError, "new directory"):
                refinement.run(self.args)
            model.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"untouched")
        self.assertEqual(before, self.snapshot())

    def test_failure_keeps_completed_frame_hashes_and_marks_manifest_incomplete(self):
        self.archive([scene(region()), scene(region())])
        before = self.snapshot()
        with patch.object(refinement, "LocalSAM3") as model:
            model.return_value.metadata = {}
            model.return_value.predict_regions.side_effect = [([], {"total_seconds": 0}), RuntimeError("synthetic failure")]
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                refinement.run(self.args)
        manifest = json.loads((self.args.output / "manifest.json").read_bytes())
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual([entry["frame_id"] for entry in manifest["frames"]], ["f0"])
        for relative, digest in manifest["frames"][0]["artifact_sha256"].items():
            self.assertEqual(file_hash(self.args.output / relative), digest)
        self.assertEqual(before, self.snapshot())

    def test_scene_loader_returns_documented_bindings(self):
        manifest, prepared = load_scene_run(self.source, self.scenes)
        self.assertEqual(manifest, self.scene_manifest)
        self.assertEqual(set(prepared[0]), {"frame", "scene", "record", "image", "image_bytes",
                                           "image_sha256", "scene_record_sha256"})

    def test_cli_defaults_are_cuda_four_threads_and_point_three(self):
        argv = ["refine_scene_regions.py", "--results", str(self.source), "--scenes", str(self.scenes),
                "--output", str(self.args.output), "--weights", str(self.weights), "--clip-dir", str(self.root)]
        with patch.object(sys, "argv", argv), patch.object(refinement, "run") as run:
            refinement.main()
        args = run.call_args.args[0]
        self.assertEqual((args.device, args.threads, args.confidence), ("cuda", 4, 0.3))
        self.assertEqual(args.clip_dir, self.root)


if __name__ == "__main__":
    unittest.main()
