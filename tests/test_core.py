import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import field_boundary_core as core


IDENTITY = [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]
INTRINSICS = [100.0, 100.0, 50.0, 40.0]


def observation(**changes):
    value = dict(frame_id="f1", evidence_group="g1", capture_time=10.0,
                 publish_time=10.1, epoch="epoch-1", points_xy_m=[[0.1, 0.1]],
                 semantic="crop", semantic_score=0.73, position_sigma_m=0.4,
                 camera_xy_m=[0.0, 0.0], geometry_source="measured_depth")
    value.update(changes)
    return value


class GuardAssertions(unittest.TestCase):
    def assert_guarded(self, value):
        self.assertIs(value["maneuver_verified"], False)
        self.assertEqual(value["authorization"], "unknown")
        self.assertNotEqual(value.get("status"), "safe")


class HeadlandTests(GuardAssertions):
    def test_unknown_without_ground_or_contact(self):
        for ground, contact in [(0, 0), (0, 1), (1, 0), (0.009, 1), (1, 0.009)]:
            with self.subTest(ground=ground, contact=contact):
                result = core.assess_headland(ground, contact, confirmed_row_ends=5)
                self.assertEqual(result["hypothesis"], "unknown")
                self.assertEqual(result["status"], "unknown")
                self.assertTrue(result["reasons"])
                self.assert_guarded(result)

    def test_contact_without_physical_row_ends_is_ambiguous(self):
        for ends in [0, 1]:
            result = core.assess_headland(0.8, 0.4, ends)
            self.assertEqual(result["hypothesis"], "headland_or_field_road")
            self.assertEqual(result["status"], "ambiguous")
            self.assert_guarded(result)

    def test_candidate_requires_multiple_confirmed_ends(self):
        result = core.assess_headland(0.01, 0.01, 2)
        self.assertEqual(result["hypothesis"], "candidate_headland")
        self.assertEqual(result["status"], "candidate")
        self.assert_guarded(result)

    def test_fov_anchors_never_upgrade(self):
        for ends in [0, 1, 2, 100]:
            result = core.assess_headland(0.9, 0.9, ends, row_end_at_fov_edge=True)
            self.assertEqual(result["status"], "ambiguous")
            self.assertIn("row_ends_at_fov_edge", result["reasons"])
            self.assert_guarded(result)

    def test_rows_continue_is_priority_counterevidence(self):
        for ground, contact, fov in [(0.8, 0.8, False), (0.8, 0.8, True), (0, 0, False)]:
            result = core.assess_headland(ground, contact, 10, True, fov)
            self.assertEqual(result["hypothesis"], "field_road_or_gap")
            self.assertEqual(result["status"], "conflicting_headland")
            self.assertEqual(result["reasons"], ["rows_continue"])
            self.assert_guarded(result)

    def test_invalid_fractions_and_counts(self):
        for bad in [-0.01, 1.01, math.nan, math.inf, -math.inf]:
            for args in [(bad, 0.5), (0.5, bad)]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    core.assess_headland(*args)
        with self.assertRaises(ValueError):
            core.assess_headland(0.5, 0.5, -1)

    def test_invalid_types_do_not_silently_coerce(self):
        for changes in [dict(open_ground_fraction="0.5"), dict(crop_contact_fraction=True),
                        dict(confirmed_row_ends=2.5), dict(confirmed_row_ends=2**100),
                        dict(rows_continue="false"), dict(row_end_at_fov_edge=0)]:
            args = dict(open_ground_fraction=0.5, crop_contact_fraction=0.5)
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                core.assess_headland(**args)


class RegionSupportTests(GuardAssertions):
    def assert_support(self, result, *, status, inside, outside, coverage,
                       edge=False, threshold=0.15):
        area = inside + outside
        self.assertEqual(result, dict(
            status=status, mask_area_pixels=area, inside_pixels=inside, outside_pixels=outside,
            outside_fraction=outside / area if area else None, box_coverage_fraction=coverage,
            touches_image_edge=edge, max_outside_fraction=threshold, semantic_verified=False,
            ego_vehicle_verified=False, maneuver_verified=False, authorization="unknown"))
        for key in ["mask_area_pixels", "inside_pixels", "outside_pixels"]:
            self.assertIs(type(result[key]), int)
        self.assertIs(result["touches_image_edge"], edge)
        self.assertIs(result["semantic_verified"], False)
        self.assertIs(result["ego_vehicle_verified"], False)
        self.assert_guarded(result)

    def test_counts_use_actual_pixels_not_mask_bounding_box(self):
        mask = np.zeros((6, 8), dtype=bool)
        for y, x in [(1, 1), (1, 3), (2, 1), (3, 3), (1, 6), (4, 2)]:
            mask[y, x] = True
        result = core.assess_region_support(mask, [1, 1, 5, 4])
        self.assert_support(result, status="outside_proposal", inside=4, outside=2, coverage=4 / 12)

    def test_sparse_support_does_not_require_proposal_to_be_pixel_truth(self):
        mask = np.zeros((7, 9), dtype=bool)
        mask[3, 4] = True
        result = core.assess_region_support(mask=mask, box_xyxy=[1, 1, 8, 6])
        self.assert_support(result, status="spatially_consistent", inside=1, outside=0, coverage=1 / 35)

    def test_box_is_xy_and_half_open(self):
        mask = np.zeros((6, 9), dtype=bool)
        for y, x in [(1, 2), (3, 5), (1, 6), (4, 2), (1, 1), (0, 2)]:
            mask[y, x] = True
        result = core.assess_region_support(mask, (2, 1, 6, 4))
        self.assert_support(result, status="outside_proposal", inside=2, outside=4,
                            coverage=2 / 12, edge=True)

    def test_empty_mask_has_priority_and_no_outside_fraction(self):
        mask = np.zeros((5, 7), dtype=bool)
        for box in [[0, 0, 7, 5], [1, 1, 6, 4]]:
            for threshold in [0, 0.15, math.nextafter(1.0, 0.0)]:
                with self.subTest(box=box, threshold=threshold):
                    result = core.assess_region_support(mask, box, threshold)
                    self.assert_support(result, status="empty_mask", inside=0, outside=0,
                                        coverage=0.0, threshold=threshold)
                    self.assertIsNone(result["outside_fraction"])

    def test_default_threshold_is_inclusive(self):
        mask = np.zeros((6, 9), dtype=bool)
        mask[1:5, 1:6] = np.arange(20).reshape(4, 5) < 17
        mask[1:4, 7] = True
        result = core.assess_region_support(mask, [1, 1, 6, 5])
        self.assert_support(result, status="spatially_consistent", inside=17, outside=3, coverage=17 / 20)
        self.assertEqual(result["outside_fraction"], result["max_outside_fraction"])

    def test_custom_threshold_equality_and_adjacent_values(self):
        mask = np.zeros((5, 7), dtype=bool)
        for y, x in [(1, 1), (1, 2), (2, 1), (2, 4)]:
            mask[y, x] = True
        for threshold, status in [(math.nextafter(0.25, 0.0), "outside_proposal"),
                                  (0.25, "spatially_consistent"),
                                  (math.nextafter(0.25, 1.0), "spatially_consistent")]:
            with self.subTest(threshold=threshold):
                result = core.assess_region_support(mask, [1, 1, 3, 3], max_outside_fraction=threshold)
                self.assert_support(result, status=status, inside=3, outside=1,
                                    coverage=3 / 4, threshold=threshold)

    def test_zero_threshold_rejects_any_outside_pixel(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = core.assess_region_support(mask, [1, 1, 4, 4], 0)
        self.assert_support(result, status="spatially_consistent", inside=1, outside=0,
                            coverage=1 / 9, threshold=0)
        mask[1, 4] = True
        result = core.assess_region_support(mask, [1, 1, 4, 4], 0)
        self.assert_support(result, status="outside_proposal", inside=1, outside=1,
                            coverage=1 / 9, threshold=0, edge=True)

    def test_all_outside_never_becomes_spatially_consistent(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[2, 5] = True
        for threshold in [0.15, math.nextafter(1.0, 0.0)]:
            result = core.assess_region_support(mask, [1, 1, 3, 4], threshold)
            self.assert_support(result, status="outside_proposal", inside=0, outside=1,
                                coverage=0.0, threshold=threshold)

    def test_each_image_edge_and_corner_uses_true_pixels(self):
        for y, x in [(0, 3), (4, 3), (2, 0), (2, 6), (0, 0), (0, 6), (4, 0), (4, 6)]:
            with self.subTest(y=y, x=x):
                mask = np.zeros((5, 7), dtype=bool)
                mask[y, x] = True
                result = core.assess_region_support(mask, [0, 0, 7, 5])
                self.assert_support(result, status="spatially_consistent", inside=1, outside=0,
                                    coverage=1 / 35, edge=True)
                result = core.assess_region_support(mask, [1, 1, 6, 4])
                self.assert_support(result, status="outside_proposal", inside=0, outside=1,
                                    coverage=0.0, edge=True)

    def test_proposal_touching_edge_does_not_mean_mask_touches_edge(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[2, 3] = True
        result = core.assess_region_support(mask, [0, 0, 7, 5])
        self.assert_support(result, status="spatially_consistent", inside=1, outside=0, coverage=1 / 35)

    def test_full_frame_mask_box_coverage_is_not_iou(self):
        mask = np.ones((5, 7), dtype=bool)
        result = core.assess_region_support(mask, [1, 1, 6, 4])
        self.assert_support(result, status="outside_proposal", inside=15, outside=20,
                            coverage=1.0, edge=True)

    def test_single_pixel_row_and_column_masks(self):
        for height, width in [(1, 1), (1, 7), (5, 1)]:
            with self.subTest(shape=(height, width)):
                mask = np.zeros((height, width), dtype=bool)
                mask[height // 2, width // 2] = True
                result = core.assess_region_support(mask, [0, 0, width, height])
                self.assert_support(result, status="spatially_consistent", inside=1, outside=0,
                                    coverage=1 / (height * width), edge=True)

    def test_read_only_c_contiguous_view_is_accepted_without_mutation(self):
        base = np.zeros((8, 9), dtype=bool)
        base[3, 4] = True
        before = base.copy()
        mask = base[1:7]
        mask.setflags(write=False)
        self.assertTrue(mask.flags.c_contiguous)
        result = core.assess_region_support(mask, [1, 1, 8, 5])
        self.assert_support(result, status="spatially_consistent", inside=1, outside=0, coverage=1 / 28)
        self.assertFalse(mask.flags.writeable)
        np.testing.assert_array_equal(base, before)

    def test_noncanonical_true_bool_bytes_are_counted_without_normalizing(self):
        storage = np.array([[0, 2, 0], [255, 1, 128]], dtype=np.uint8)
        mask = storage.view(np.bool_)
        mask.setflags(write=False)
        before = storage.tobytes()
        result = core.assess_region_support(mask, [1, 0, 3, 2], 0.25)
        self.assert_support(result, status="spatially_consistent", inside=3, outside=1,
                            coverage=3 / 4, threshold=0.25, edge=True)
        self.assertEqual(storage.tobytes(), before)
        self.assertFalse(mask.flags.writeable)

    def test_inputs_unchanged_on_success_and_rejection(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[2:4, 2:4] = True
        mask[1, 6] = True
        before = (mask.tobytes(), mask.shape, mask.strides, mask.dtype, mask.flags.writeable)
        for box, threshold, status in [([1, 1, 5, 5], 0.15, "outside_proposal"),
                                       ([1, 1, 5, 5], 0.2, "spatially_consistent"),
                                       ([0, 0, 8, 6], 0.15, "spatially_consistent"),
                                       ([-1, 0, 8, 6], 0.15, None),
                                       ([0, 0, 8, 6], 1.0, None)]:
            original_box = box.copy()
            with self.subTest(box=box, threshold=threshold):
                if status is None:
                    with self.assertRaises(ValueError):
                        core.assess_region_support(mask, box, threshold)
                else:
                    self.assertEqual(core.assess_region_support(mask, box, threshold)["status"], status)
                self.assertEqual(box, original_box)
                self.assertEqual(before, (mask.tobytes(), mask.shape, mask.strides,
                                          mask.dtype, mask.flags.writeable))
        mask.fill(False)
        before_bytes = mask.tobytes()
        self.assertEqual(core.assess_region_support(mask, [0, 0, 8, 6])["status"], "empty_mask")
        self.assertEqual(mask.tobytes(), before_bytes)

    def test_rejects_non_numpy_masks_without_implicit_conversion(self):
        class ArrayLike:
            def __array__(self, *args, **kwargs):
                raise AssertionError("implicit array conversion must not be attempted")

        for mask in [None, True, [[True, False]], ((True, False),),
                     memoryview(np.zeros((2, 3), dtype=bool)), ArrayLike()]:
            with self.subTest(type=type(mask)), self.assertRaises(ValueError):
                core.assess_region_support(mask, [0, 0, 1, 1])

    def test_rejects_non_bool_dtypes(self):
        for dtype in [np.uint8, np.int8, np.int64, np.float32, np.float64, np.complex128,
                      object, "U1", [("value", np.bool_)]]:
            mask = np.zeros((2, 3), dtype=dtype)
            before = mask.tobytes()
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                core.assess_region_support(mask, [0, 0, 3, 2])
            self.assertEqual(mask.tobytes(), before)

    def test_rejects_wrong_dimensions(self):
        for shape in [(), (6,), (1, 2, 3), (1, 1, 2, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                core.assess_region_support(np.zeros(shape, dtype=bool), [0, 0, 1, 1])

    def test_rejects_empty_dimensions(self):
        for shape in [(0, 0), (0, 3), (2, 0)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                core.assess_region_support(np.zeros(shape, dtype=bool), [0, 0, 1, 1])

    def test_rejects_non_c_contiguous_strides(self):
        base = np.zeros((6, 8), dtype=bool)
        for mask in [base.T, np.asfortranarray(base), base[:, ::2], base[::2, :],
                     base[::-1], base[:, ::-1], np.broadcast_to(base[:1], (6, 8))]:
            self.assertFalse(mask.flags.c_contiguous)
            with self.subTest(strides=mask.strides), self.assertRaisesRegex(ValueError, "C-contiguous"):
                core.assess_region_support(mask, [0, 0, 1, 1])

    def test_pixel_limit_is_inclusive(self):
        mask = np.zeros((4096, 4096), dtype=bool)
        mask[-1, -1] = True
        result = core.assess_region_support(mask, [0, 0, 4096, 4096])
        self.assert_support(result, status="spatially_consistent", inside=1, outside=0,
                            coverage=1 / 16777216, edge=True)

    def test_rejects_above_pixel_limit(self):
        for shape in [(1, 16777217), (4097, 4096)]:
            mask = np.zeros(shape, dtype=bool)
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "16777216"):
                core.assess_region_support(mask, [0, 0, 1, 1])

    def test_rejects_invalid_box_sequences_and_coordinate_types(self):
        mask = np.zeros((5, 7), dtype=bool)
        boxes = [None, 1, True, "0011", b"0011", {0: 0, 1: 0, 2: 1, 3: 1},
                 {0, 1, 2, 3}, (x for x in [0, 0, 1, 1]), [], [0, 0, 1],
                 [0, 0, 1, 1, 2], [[0, 0], [1, 1]], [[0], [0], [1], [1]]]
        for box in boxes:
            with self.subTest(box=box), self.assertRaises(ValueError):
                core.assess_region_support(mask, box)
        for bad in [True, False, np.bool_(True), np.bool_(False), 1.0, 1.25, "1", None,
                    math.nan, math.inf, 1 + 0j, 2**100, -(2**100)]:
            for index in range(4):
                box = [1, 1, 6, 4]
                box[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    core.assess_region_support(mask, box)

    def test_rejects_empty_reversed_and_out_of_bounds_boxes(self):
        mask = np.zeros((5, 7), dtype=bool)
        boxes = [[-1, 0, 3, 3], [0, -1, 3, 3], [0, 0, 8, 5], [0, 0, 7, 6],
                 [2, 1, 2, 4], [1, 2, 6, 2], [3, 1, 2, 4], [1, 3, 6, 2],
                 [0, 0, 0, 0], [7, 0, 8, 1], [0, 5, 1, 6], [0, 0, -1, 1],
                 [0, 0, 1, -1], [-(2**63), 0, 1, 1], [0, 0, 2**63 - 1, 1],
                 [0, 0, 1, 2**63 - 1]]
        for box in boxes:
            with self.subTest(box=box), self.assertRaises(ValueError):
                core.assess_region_support(mask, box)

    def test_rejects_invalid_thresholds_even_for_empty_masks(self):
        for filled in [False, True]:
            mask = np.full((3, 4), filled, dtype=bool)
            for threshold in [-0.01, 1, 1.01, math.nan, math.inf, -math.inf, True, False,
                              "0.15", None, [], {}, 0.15 + 0j, 10**1000]:
                with self.subTest(filled=filled, threshold=threshold), self.assertRaises(ValueError):
                    core.assess_region_support(mask, [0, 0, 4, 3], threshold)

    def test_matches_numpy_pixel_count_oracle(self):
        rng = np.random.default_rng(2026)
        for height, width in [(3, 8), (8, 3), (7, 11)]:
            for density in [0, 0.1, 0.5, 1]:
                mask = rng.random((height, width)) < density
                x0, y0 = int(rng.integers(width)), int(rng.integers(height))
                x1, y1 = int(rng.integers(x0 + 1, width + 1)), int(rng.integers(y0 + 1, height + 1))
                inside = int(np.count_nonzero(mask[y0:y1, x0:x1]))
                area = int(np.count_nonzero(mask))
                outside = area - inside
                edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
                before = mask.tobytes()
                for threshold in [0, 0.15, 0.5, math.nextafter(1.0, 0.0)]:
                    status = ("empty_mask" if not area else "spatially_consistent"
                              if inside > 0 and outside / area <= threshold else "outside_proposal")
                    with self.subTest(shape=mask.shape, density=density, threshold=threshold):
                        result = core.assess_region_support(mask, [x0, y0, x1, y1], threshold)
                        self.assert_support(result, status=status, inside=inside, outside=outside,
                                            coverage=inside / ((x1 - x0) * (y1 - y0)),
                                            edge=edge, threshold=threshold)
                        self.assertEqual(mask.tobytes(), before)


class ProjectionTests(unittest.TestCase):
    def project(self, **changes):
        args = dict(pixel_xy=[50.0, 40.0], depth_m=2.0, intrinsics=INTRINSICS,
                    world_from_camera=IDENTITY, image_size=[100, 80])
        args.update(changes)
        return core.project_depth(**args)

    def test_identity_and_optical_axis_depth(self):
        self.assertEqual(self.project(), [0.0, 0.0, 2.0])
        self.assertEqual(self.project(pixel_xy=[75, 65]), [0.5, 0.5, 2.0])
        self.assertEqual(self.project(pixel_xy=[0, 0]), [-1.0, -0.8, 2.0])
        self.assertEqual(self.project(pixel_xy=[99.999, 79.999])[2], 2.0)

    def test_camera_to_enu_rotation_translation(self):
        transform = [1, 0, 0, 10, 0, 0, 1, 20, 0, -1, 0, 30, 0, 0, 0, 1]
        self.assertEqual(self.project(pixel_xy=[75, 65], world_from_camera=transform), [10.5, 22.0, 29.5])
        angle = 0.43
        c, s = math.cos(angle), math.sin(angle)
        yaw = [c, -s, 0, 1, s, c, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1]
        result = self.project(pixel_xy=[75, 65], world_from_camera=yaw)
        for actual, expected in zip(result, [1 + 0.5 * (c - s), 2 + 0.5 * (s + c), 5]):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_sample_returns_none(self):
        for depth in [0, -1, math.nan, math.inf, -math.inf, 30.001]:
            with self.subTest(depth=depth):
                self.assertIsNone(self.project(depth_m=depth))
        for pixel in [[-0.001, 0], [0, -0.001], [100, 0], [0, 80], [math.nan, 0], [0, math.inf]]:
            with self.subTest(pixel=pixel):
                self.assertIsNone(self.project(pixel_xy=pixel))

    def test_range_is_euclidean_not_axis_depth(self):
        self.assertEqual(self.project(depth_m=30), [0, 0, 30])
        self.assertIsNone(self.project(depth_m=30, pixel_xy=[99, 79]))
        self.assertIsNone(self.project(depth_m=2, pixel_xy=[75, 65], max_range_m=2.1))
        self.assertIsNotNone(self.project(depth_m=40, max_range_m=50))
        self.assertIsNone(self.project(intrinsics=[1e-300, 100, 50, 40], pixel_xy=[99, 40]))

    def test_intrinsics_and_configuration_validation(self):
        for intrinsics in [[0, 100, 50, 40], [-1, 100, 50, 40], [100, 0, 50, 40],
                           [100, math.nan, 50, 40], [100, 100, math.inf, 40], [1, 2, 3], "bad"]:
            with self.subTest(intrinsics=intrinsics), self.assertRaises(ValueError):
                self.project(intrinsics=intrinsics)
        for size in [[0, 80], [100, -1], [100.0, 80], [True, 80], [100], [2**40, 80]]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.project(image_size=size)
        for distance in [0, -1, math.nan, math.inf, "30", True, None]:
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.project(max_range_m=distance)
        for depth in ["2", True, None, 10**1000]:
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.project(depth_m=depth)

    def test_transform_validation_precedes_sample_rejection(self):
        for index, replacement in [(0, 2), (0, -1), (1, 0.1), (12, 0.1), (15, 0.9),
                                   (3, math.inf), (7, math.nan)]:
            transform = IDENTITY.copy()
            transform[index] = replacement
            with self.subTest(index=index, replacement=replacement), self.assertRaises(ValueError):
                self.project(world_from_camera=transform, depth_m=0)
        for transform in [IDENTITY[:-1], [IDENTITY[:4]] * 4, "bad"]:
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                self.project(world_from_camera=transform)
        for pixel in [[1], [1, 2, 3], "50,40"]:
            with self.subTest(pixel=pixel), self.assertRaises(ValueError):
                self.project(pixel_xy=pixel)


class EvidenceTests(GuardAssertions):
    def setUp(self):
        self.map = core.EvidenceMap("field-1", "epoch-1")

    def cell(self, now=10.1):
        snapshot = self.map.snapshot(now)
        self.assert_guarded(snapshot)
        self.assertEqual(snapshot["unexplored"], "unknown")
        self.assertFalse(snapshot["consistent_is_authorization"])
        cell = snapshot["cells"][0]
        self.assert_guarded(cell)
        return cell

    def test_constructor_guards(self):
        bad_configs = [dict(map_id=""), dict(epoch=""), dict(map_id=" " * 3), dict(epoch="x" * 257),
                       dict(cell_size_m=0), dict(cell_size_m=-1), dict(cell_size_m=math.nan),
                       dict(cell_size_m=math.inf), dict(cell_size_m=1000001),
                       dict(ttl_s=0), dict(ttl_s=-1), dict(ttl_s=math.inf), dict(ttl_s=math.nan),
                       dict(min_views=0), dict(min_views=257), dict(min_views=1.5), dict(min_views=True),
                       dict(min_baseline_m=-1), dict(min_baseline_m=math.nan), dict(min_baseline_m=math.inf),
                       dict(max_cells=0), dict(max_cells=-1), dict(max_cells=1000001),
                       dict(max_cells=1.5), dict(max_cells=True), dict(max_cells=2**100),
                       dict(map_id=123), dict(epoch=None), dict(cell_size_m=True),
                       dict(ttl_s="5"), dict(min_baseline_m=None)]
        for changes in bad_configs:
            args = dict(map_id="field", epoch="epoch")
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                core.EvidenceMap(**args)

    def test_empty_map_stays_unknown(self):
        result = self.map.snapshot(0)
        self.assertEqual(result["cells"], [])
        self.assertEqual(result["status_counts"], dict(candidate=0, consistent=0, conflict=0, expired=0))
        self.assertEqual(result["unexplored"], "unknown")
        self.assertIsNone(result["last_capture_time"])
        self.assertEqual(result["coordinate_frame"], "world_enu_m")
        self.assert_guarded(result)
        rejected = self.map.update(observation(points_xy_m=[]))
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["reason"], "empty_points")
        self.assertEqual(result, self.map.snapshot(0))

    def test_first_evidence_records_raw_source(self):
        result = self.map.update(observation())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["added_sources"], 1)
        self.assert_guarded(result)
        cell = self.cell()
        self.assertEqual(cell["cell_index"], [0, 0])
        self.assertEqual(cell["center_xy_m"], [0.125, 0.125])
        self.assertEqual(cell["status"], "candidate")
        self.assertEqual(cell["semantic"], "crop")
        self.assertEqual(cell["semantic_labels"], ["crop"])
        self.assertEqual(cell["independent_views"], 1)
        self.assertEqual(cell["position_sigma_m"], 0.4)
        source = cell["sources"][0]
        for field in ["frame_id", "evidence_group", "capture_time", "publish_time", "semantic",
                      "semantic_score", "position_sigma_m", "camera_xy_m", "geometry_source"]:
            self.assertEqual(source[field], observation()[field])
        self.assertTrue(source["active"])
        self.assertTrue(source["independent_view"])

    def test_repeated_points_count_once_per_cell(self):
        result = self.map.update(observation(points_xy_m=[[0.1, 0.1]] * 1000 + [[0.2, 0.2]]))
        self.assertEqual(result["touched_cells"], 1)
        self.assertEqual(result["added_sources"], 1)
        self.assertEqual(self.cell()["independent_views"], 1)

    def test_same_frame_replay_does_not_refresh(self):
        self.map.update(observation())
        before = self.cell()
        result = self.map.update(observation(publish_time=12, semantic_score=0.99, position_sigma_m=0.01))
        self.assertEqual(result["reason"], "duplicate_evidence")
        self.assertEqual(result["added_sources"], 0)
        self.assertEqual(result["duplicate_cells"], 1)
        self.assertEqual(before, self.cell(12))
        self.assertEqual(self.cell(15.01)["status"], "expired")

    def test_same_track_new_frame_and_camera_are_not_new_evidence(self):
        self.map.update(observation())
        before = self.cell()
        result = self.map.update(observation(frame_id="f2", capture_time=12, publish_time=12,
                                             camera_xy_m=[10, 0], semantic_score=0.99, position_sigma_m=0.01))
        self.assertTrue(result["accepted"])
        self.assertEqual(before, self.cell(12))
        self.assertEqual(self.cell(15.01)["status"], "expired")

    def test_same_frame_different_groups_cannot_count_twice(self):
        self.map.update(observation())
        self.map.update(observation(evidence_group="other-prompt", camera_xy_m=[10, 0]))
        cell = self.cell()
        self.assertEqual(cell["active_groups"], 2)
        self.assertEqual(cell["independent_views"], 1)
        self.assertEqual(cell["status"], "candidate")

    def test_independent_baseline_and_distinct_groups(self):
        self.map.update(observation())
        self.map.update(observation(frame_id="f2", evidence_group="g2", camera_xy_m=[0.499, 0]))
        self.assertEqual(self.cell()["independent_views"], 1)
        self.map.update(observation(frame_id="f3", evidence_group="g3", camera_xy_m=[0.5, 0], position_sigma_m=0.01))
        cell = self.cell()
        self.assertEqual(cell["independent_views"], 2)
        self.assertEqual(cell["status"], "consistent")
        self.assertEqual(cell["position_sigma_m"], 0.4)

    def test_baseline_is_pairwise_to_all_counted_views(self):
        self.map = core.EvidenceMap("field-1", "epoch-1", min_views=3)
        for index, x in enumerate([0, 0.5, 0.9]):
            self.map.update(observation(frame_id=f"f{index}", evidence_group=f"g{index}", camera_xy_m=[x, 0]))
        self.assertEqual(self.cell()["independent_views"], 2)
        self.assertEqual(self.cell()["status"], "candidate")
        self.map.update(observation(frame_id="f4", evidence_group="g4", camera_xy_m=[1, 0]))
        self.assertEqual(self.cell()["status"], "consistent")

    def test_same_group_conflict_preserved_without_multiview(self):
        self.map.update(observation())
        self.map.update(observation(frame_id="f2", semantic="open_ground", capture_time=12, publish_time=12,
                                    camera_xy_m=[5, 0], position_sigma_m=0.8, semantic_score=0.32))
        cell = self.cell(12)
        self.assertEqual(cell["semantic"], "conflict")
        self.assertEqual(cell["semantic_labels"], ["crop", "open_ground"])
        self.assertEqual(cell["status"], "conflict")
        self.assertEqual(cell["independent_views"], 1)
        self.assertEqual(cell["active_groups"], 1)
        self.assertEqual(cell["position_sigma_m"], 0.8)
        self.assertEqual([s["semantic_score"] for s in cell["sources"]], [0.73, 0.32])
        self.assertEqual(cell["sources"][1]["camera_xy_m"], [5, 0])
        self.assertEqual(cell["sources"][1]["view_anchor_xy_m"], [0, 0])
        self.map.update(observation(frame_id="f3", semantic="open_ground", capture_time=14, publish_time=14))
        self.assertEqual(cell, self.cell(14))
        self.assertEqual(self.cell(15.01)["status"], "expired")

    def test_cross_group_conflict_overrides_consistency(self):
        self.map.update(observation())
        self.map.update(observation(frame_id="f2", evidence_group="g2", semantic="field_road",
                                    camera_xy_m=[1, 0], capture_time=12, publish_time=12))
        self.assertEqual(self.cell(12)["status"], "conflict")
        self.assertEqual(self.cell(12)["independent_views"], 2)
        current = self.cell(15.01)
        self.assertEqual(current["status"], "candidate")
        self.assertEqual(current["semantic"], "field_road")
        self.assertEqual(current["independent_views"], 1)
        self.assertFalse(current["sources"][0]["active"])

    def test_unknown_never_becomes_consistent_or_free(self):
        for index in range(3):
            self.map.update(observation(frame_id=f"f{index}", evidence_group=f"g{index}",
                                        semantic="unknown", camera_xy_m=[index, 0]))
        cell = self.cell()
        self.assertEqual(cell["status"], "candidate")
        self.assertEqual(cell["semantic"], "unknown")
        self.assertEqual(self.cell(20)["status"], "expired")
        self.assertEqual(self.cell(20)["semantic_labels"], [])

    def test_ttl_uses_capture_not_publish_and_is_inclusive(self):
        self.map.update(observation(publish_time=15))
        self.assertEqual(self.cell(15)["status"], "candidate")
        expired = self.cell(15.0001)
        self.assertEqual(expired["status"], "expired")
        self.assertEqual(expired["semantic"], "unknown")
        self.assertEqual(expired["independent_views"], 0)
        self.assertFalse(expired["sources"][0]["active"])

    def test_future_snapshot_cannot_resurrect_expired_evidence(self):
        self.map.update(observation())
        self.assertEqual(self.cell(100)["status"], "expired")
        with self.assertRaises(ValueError):
            self.map.snapshot(11)
        rejected = self.map.update(observation(frame_id="f2", evidence_group="g2", capture_time=12, publish_time=12))
        self.assertEqual(rejected["reason"], "capture_expired")
        replay = self.map.update(observation(frame_id="f2", capture_time=100, publish_time=100))
        self.assertEqual(replay["reason"], "duplicate_evidence")
        self.assertEqual(self.cell(100)["status"], "expired")
        self.map.update(observation(frame_id="fresh", evidence_group="fresh", capture_time=101, publish_time=101))
        self.assertEqual(self.cell(101)["status"], "candidate")
        self.assertEqual(self.cell(101)["independent_views"], 1)

    def test_stale_same_group_conflict_does_not_revive_group(self):
        self.map.update(observation())
        self.cell(20)
        result = self.map.update(observation(frame_id="later", semantic="open_ground", capture_time=20, publish_time=20))
        self.assertTrue(result["accepted"])
        cell = self.cell(20)
        self.assertEqual(cell["status"], "expired")
        self.assertEqual(len(cell["sources"]), 2)
        self.assertFalse(cell["sources"][1]["active"])

    def test_capture_order_rejected_without_mutation(self):
        self.map.update(observation(capture_time=12, publish_time=12))
        before = self.map.snapshot(12)
        result = self.map.update(observation(frame_id="older", evidence_group="older", capture_time=11, publish_time=12))
        self.assertEqual(result["reason"], "out_of_order_capture")
        self.assertFalse(result["accepted"])
        self.assertEqual(before, self.map.snapshot(12))

    def test_epoch_is_fixed_and_new_map_is_isolated(self):
        self.map.update(observation())
        before = self.map.snapshot(10.1)
        result = self.map.update(observation(epoch="epoch-2", capture_time=11, publish_time=11))
        self.assertEqual(result["reason"], "epoch_mismatch")
        self.assertEqual(before, self.map.snapshot(10.1))
        other = core.EvidenceMap("field-1", "epoch-2")
        self.assertTrue(other.update(observation(epoch="epoch-2"))["accepted"])
        self.assertEqual(other.snapshot(10.1)["epoch"], "epoch-2")

    def test_invalid_observation_values_are_atomic_rejections(self):
        self.map.update(observation())
        before = self.map.snapshot(10.1)
        cases = [({"frame_id": ""}, "invalid_frame_id"),
                 ({"evidence_group": "x" * 257}, "invalid_evidence_group"),
                 ({"capture_time": math.nan}, "nonfinite_timestamp"),
                 ({"publish_time": math.inf}, "nonfinite_timestamp"),
                 ({"capture_time": 11, "publish_time": 10.1}, "capture_after_publish"),
                 ({"publish_time": 16}, "capture_expired"),
                 ({"semantic_score": -0.1}, "invalid_semantic_score"),
                 ({"semantic_score": 1.1}, "invalid_semantic_score"),
                 ({"semantic_score": math.nan}, "invalid_semantic_score"),
                 ({"semantic_score": math.inf}, "invalid_semantic_score"),
                 ({"position_sigma_m": -1}, "invalid_position_sigma_m"),
                 ({"position_sigma_m": math.inf}, "invalid_position_sigma_m"),
                 ({"position_sigma_m": math.nan}, "invalid_position_sigma_m"),
                 ({"semantic": ""}, "invalid_semantic"),
                 ({"semantic": "x" * 129}, "invalid_semantic"),
                 ({"camera_xy_m": [math.nan, 0]}, "invalid_camera_xy_m"),
                 ({"camera_xy_m": [1000001, 0]}, "invalid_camera_xy_m"),
                 ({"points_xy_m": [[0.1, 0.1], [math.inf, 1]]}, "invalid_point_xy_m"),
                 ({"points_xy_m": [[0.1, 0.1], [1000001, 1]]}, "invalid_point_xy_m"),
                 ({"points_xy_m": []}, "empty_points")]
        for changes, reason in cases:
            with self.subTest(changes=changes):
                result = self.map.update(observation(**changes))
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["added_sources"], 0)
                self.assertEqual(before, self.map.snapshot(10.1))

    def test_only_measured_depth_geometry_is_accepted(self):
        before = self.map.snapshot(0)
        for geometry in ["", "plane", "ground_plane", "monocular", "rowmap_anchor", "predicted_depth"]:
            with self.subTest(geometry=geometry):
                result = self.map.update(observation(geometry_source=geometry))
                self.assertEqual(result["reason"], "geometry_not_measured_depth")
                self.assertEqual(before, self.map.snapshot(0))

    def test_malformed_input_types_and_missing_fields_are_atomic(self):
        self.map.update(observation())
        before = self.map.snapshot(10.1)
        invalid = [None, [], "bad", observation(points_xy_m=[[0.1, 0.1], [1]]),
                   observation(points_xy_m="bad"), observation(camera_xy_m=[0]),
                   observation(frame_id=123), observation(evidence_group=123),
                   observation(semantic_score="0.8"), observation(semantic_score=True),
                   observation(position_sigma_m=None), observation(points_xy_m=[["x", 1]])]
        for key in observation():
            missing = observation()
            del missing[key]
            invalid.append(missing)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.map.update(value)
            self.assertEqual(before, self.map.snapshot(10.1))

    def test_max_cells_rejection_does_not_partially_add_conflict(self):
        self.map = core.EvidenceMap("field-1", "epoch-1", max_cells=1)
        self.map.update(observation())
        before = self.map.snapshot(10.1)
        result = self.map.update(observation(semantic="open_ground", points_xy_m=[[0.1, 0.1], [0.4, 0.1]]))
        self.assertEqual(result["reason"], "max_cells_exceeded")
        self.assertEqual(before, self.map.snapshot(10.1))
        self.map.snapshot(20)
        result = self.map.update(observation(frame_id="new", evidence_group="new", capture_time=20, publish_time=20,
                                             points_xy_m=[[0.4, 0.1]]))
        self.assertEqual(result["reason"], "max_cells_exceeded")

    def test_point_count_and_cell_index_limits(self):
        before = self.map.snapshot(0)
        with self.assertRaisesRegex(ValueError, "too_many_points"):
            self.map.update(observation(points_xy_m=[[0, 0]] * 100001))
        self.assertEqual(before, self.map.snapshot(0))
        tiny = core.EvidenceMap("field", "epoch-1", cell_size_m=1e-12)
        result = tiny.update(observation(points_xy_m=[[1, 0]]))
        self.assertEqual(result["reason"], "cell_index_out_of_range")
        self.assertEqual(tiny.snapshot(0)["cells"], [])
        self.assertTrue(self.map.update(observation(points_xy_m=[[1000000, -1000000]]))["accepted"])

    def test_negative_coordinates_use_floor(self):
        self.map.update(observation(points_xy_m=[[-0.001, -0.25], [0, 0]]))
        cells = self.map.snapshot(10.1)["cells"]
        self.assertEqual([cell["cell_index"] for cell in cells], [[-1, -1], [0, 0]])
        self.assertEqual(cells[0]["center_xy_m"], [-0.125, -0.125])

    def test_per_cell_group_limit_atomic_after_staging_new_cell(self):
        self.map = core.EvidenceMap("field-1", "epoch-1", max_cells=2)
        for index in range(256):
            result = self.map.update(observation(frame_id=f"f{index}", evidence_group=f"g{index}"))
            self.assertTrue(result["accepted"])
        before = self.map.snapshot(10.1)
        result = self.map.update(observation(frame_id="overflow", evidence_group="overflow",
                                             points_xy_m=[[-0.1, 0.1], [0.1, 0.1]]))
        self.assertEqual(result["reason"], "max_groups_per_cell_exceeded")
        self.assertEqual(before, self.map.snapshot(10.1))
        self.assertEqual(self.map.update(observation(frame_id="f0", evidence_group="g0"))["reason"], "duplicate_evidence")

    def test_per_group_label_limit_atomic(self):
        for index in range(16):
            self.assertTrue(self.map.update(observation(semantic=f"label-{index}"))["accepted"])
        before = self.map.snapshot(10.1)
        result = self.map.update(observation(semantic="overflow", points_xy_m=[[-0.1, 0.1], [0.1, 0.1]]))
        self.assertEqual(result["reason"], "max_labels_per_group_exceeded")
        self.assertEqual(before, self.map.snapshot(10.1))

    def test_snapshot_validation_and_detached_results(self):
        self.map.update(observation())
        original = self.map.snapshot(10.1)
        changed = copy.deepcopy(original)
        changed["cells"][0]["sources"][0]["semantic"] = "changed"
        changed["cells"].clear()
        self.assertEqual(original, self.map.snapshot(10.1))
        for now in [math.nan, math.inf, -math.inf, 10.0, None, "11", True, 10**1000]:
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.map.snapshot(now)
            self.assertEqual(original, self.map.snapshot(10.1))

    def test_zero_score_is_raw_not_a_safety_probability(self):
        self.map = core.EvidenceMap("field-1", "epoch-1", min_views=1)
        self.map.update(observation(semantic_score=0, position_sigma_m=0))
        cell = self.cell()
        self.assertEqual(cell["sources"][0]["semantic_score"], 0)
        self.assertEqual(cell["position_sigma_m"], 0)
        self.assertEqual(cell["status"], "consistent")
        self.assert_guarded(cell)
        self.assertEqual(self.map.snapshot(10.1)["score_semantics"], "raw_model_score_not_safety_probability")


if __name__ == "__main__":
    unittest.main(verbosity=2)
