#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "field_boundary/core.hpp"

namespace py = pybind11;
namespace fb = field_boundary;

namespace {

py::dict guarded() {
    py::dict result;
    result["maneuver_verified"] = false;
    result["authorization"] = "unknown";
    return result;
}

double number(py::handle value, const std::string& name) {
    if (PyBool_Check(value.ptr())) throw py::value_error(name + " must be a number, not bool");
    try {
        return py::cast<double>(value);
    } catch (const py::cast_error&) {
        throw py::value_error(name + " must be a number");
    }
}

std::int64_t integer(py::handle value, const std::string& name) {
    if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
        throw py::value_error(name + " must be an integer");
    try {
        return py::cast<std::int64_t>(value);
    } catch (const py::cast_error&) {
        throw py::value_error(name + " is outside the supported integer range");
    }
}

std::string text(py::handle value, const std::string& name) {
    if (!py::isinstance<py::str>(value)) throw py::value_error(name + " must be a string");
    return py::cast<std::string>(value);
}

bool boolean(py::handle value, const std::string& name) {
    if (!PyBool_Check(value.ptr())) throw py::value_error(name + " must be a bool");
    return value.ptr() == Py_True;
}

py::sequence sequence(py::handle value, const std::string& name) {
    if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value) ||
        py::isinstance<py::bytes>(value))
        throw py::value_error(name + " must be a sequence");
    return py::reinterpret_borrow<py::sequence>(value);
}

template <std::size_t N>
std::array<double, N> numbers(py::handle value, const std::string& name) {
    const auto values = sequence(value, name);
    if (values.size() != N) throw py::value_error(name + " must have length " + std::to_string(N));
    std::array<double, N> result{};
    for (std::size_t i = 0; i < N; ++i) result[i] = number(values[i], name);
    return result;
}

py::handle field(const py::dict& input, const char* name) {
    if (!input.contains(name)) throw py::value_error(std::string("missing observation field: ") + name);
    return input[name];
}

fb::Observation observation_from_python(py::handle value) {
    if (!py::isinstance<py::dict>(value)) throw py::value_error("observation must be a dict");
    const auto input = py::reinterpret_borrow<py::dict>(value);
    fb::Observation result;
    result.frame_id = text(field(input, "frame_id"), "frame_id");
    result.evidence_group = text(field(input, "evidence_group"), "evidence_group");
    result.capture_time = number(field(input, "capture_time"), "capture_time");
    result.publish_time = number(field(input, "publish_time"), "publish_time");
    result.epoch = text(field(input, "epoch"), "epoch");
    result.semantic = text(field(input, "semantic"), "semantic");
    result.semantic_score = number(field(input, "semantic_score"), "semantic_score");
    result.position_sigma_m = number(field(input, "position_sigma_m"), "position_sigma_m");
    result.camera = numbers<2>(field(input, "camera_xy_m"), "camera_xy_m");
    result.geometry_source = text(field(input, "geometry_source"), "geometry_source");
    const auto points = sequence(field(input, "points_xy_m"), "points_xy_m");
    if (points.size() > fb::EvidenceMap::max_points_per_observation)
        throw py::value_error("too_many_points: points_xy_m is limited to 100000 points");
    result.points.reserve(points.size());
    for (const auto point : points) result.points.push_back(numbers<2>(point, "points_xy_m point"));
    return result;
}

py::dict update_to_python(const fb::EvidenceMap& map, const fb::UpdateResult& update) {
    auto result = guarded();
    result["map_id"] = map.map_id();
    result["epoch"] = map.epoch();
    result["unexplored"] = "unknown";
    result["accepted"] = update.accepted;
    result["reason"] = update.reason;
    result["touched_cells"] = update.touched_cells;
    result["added_cells"] = update.added_cells;
    result["added_sources"] = update.added_sources;
    result["duplicate_cells"] = update.duplicate_cells;
    result["total_cells"] = update.total_cells;
    result["total_sources"] = update.total_sources;
    result["watermark_time"] = map.watermark();
    return result;
}

py::dict snapshot_to_python(fb::EvidenceMap& map, double now) {
    map.advance_snapshot(now);
    auto result = guarded();
    result["map_id"] = map.map_id();
    result["epoch"] = map.epoch();
    result["now"] = now;
    result["watermark_time"] = map.watermark();
    result["last_capture_time"] = map.last_capture();
    result["coordinate_frame"] = "world_enu_m";
    result["unexplored"] = "unknown";
    result["score_semantics"] = "raw_model_score_not_safety_probability";
    result["consistent_is_authorization"] = false;
    result["cell_size_m"] = map.cell_size();
    result["ttl_s"] = map.ttl();
    result["min_views"] = map.min_views();
    result["min_baseline_m"] = map.baseline();
    result["max_cells"] = map.max_cells();
    result["total_cells"] = map.cells().size();
    result["total_sources"] = map.source_count();
    py::dict limits;
    limits["max_points_per_observation"] = fb::EvidenceMap::max_points_per_observation;
    limits["max_groups_per_cell"] = fb::EvidenceMap::max_groups_per_cell;
    limits["max_labels_per_group"] = fb::EvidenceMap::max_labels_per_group;
    limits["max_total_sources"] = fb::EvidenceMap::max_total_sources;
    limits["max_abs_coordinate_m"] = fb::EvidenceMap::max_abs_coordinate_m;
    limits["max_abs_cell_index"] = fb::EvidenceMap::max_abs_cell_index;
    result["limits"] = limits;
    py::list cells;
    py::dict counts;
    for (const auto* status : {"candidate", "consistent", "conflict", "expired"}) counts[status] = 0;
    for (const auto& entry : map.cells()) {
        const auto& key = entry.first;
        const auto& cell = entry.second;
        const auto state = map.state(cell);
        auto value = guarded();
        value["cell_index"] = key;
        value["center_xy_m"] = fb::XY{(key[0] + 0.5) * map.cell_size(), (key[1] + 0.5) * map.cell_size()};
        value["status"] = state.status;
        value["semantic"] = state.semantic;
        value["semantic_labels"] = state.labels;
        value["independent_views"] = state.counted_groups.size();
        value["active_groups"] = state.active_groups;
        value["position_sigma_m"] = cell.position_sigma_m;
        py::list sources;
        for (std::size_t group_index = 0; group_index < cell.groups.size(); ++group_index) {
            const auto& group = cell.groups[group_index];
            const bool counted = std::find(state.counted_groups.begin(), state.counted_groups.end(), group_index)
                                 != state.counted_groups.end();
            for (std::size_t source_index = 0; source_index < group.sources.size(); ++source_index) {
                const auto& source = group.sources[source_index];
                py::dict record;
                record["frame_id"] = source.frame_id;
                record["evidence_group"] = group.id;
                record["capture_time"] = source.capture_time;
                record["publish_time"] = source.publish_time;
                record["group_capture_time"] = group.capture_time;
                record["semantic"] = source.semantic;
                record["semantic_score"] = source.semantic_score;
                record["position_sigma_m"] = source.position_sigma_m;
                record["camera_xy_m"] = source.camera;
                record["view_anchor_xy_m"] = group.camera;
                record["geometry_source"] = "measured_depth";
                record["active"] = map.active(group);
                record["independent_view"] = counted && source_index == 0;
                sources.append(record);
            }
        }
        value["sources"] = sources;
        cells.append(value);
        const py::str status(state.status);
        counts[status] = py::cast<std::size_t>(counts[status]) + 1;
    }
    result["cells"] = cells;
    result["status_counts"] = counts;
    return result;
}

}

PYBIND11_MODULE(field_boundary_core, module) {
    module.doc() = "Bounded ENU semantic evidence, never a maneuver or free-space authorization.";
    module.def("assess_headland", [](py::handle open_ground_fraction, py::handle crop_contact_fraction,
                                    py::handle confirmed_row_ends, py::handle rows_continue, py::handle row_end_at_fov_edge) {
        const auto ends = integer(confirmed_row_ends, "confirmed_row_ends");
        fb::require(ends >= 0 && ends <= std::numeric_limits<int>::max(),
                    "confirmed_row_ends must be a nonnegative int32 value");
        const auto assessment = fb::assess_headland(number(open_ground_fraction, "open_ground_fraction"),
            number(crop_contact_fraction, "crop_contact_fraction"), static_cast<int>(ends),
            boolean(rows_continue, "rows_continue"), boolean(row_end_at_fov_edge, "row_end_at_fov_edge"));
        auto result = guarded();
        result["hypothesis"] = assessment.hypothesis;
        result["status"] = assessment.status;
        result["reasons"] = assessment.reasons;
        return result;
    }, py::arg("open_ground_fraction"), py::arg("crop_contact_fraction"), py::arg("confirmed_row_ends") = 0,
       py::arg("rows_continue") = false, py::arg("row_end_at_fov_edge") = false,
       "Assess contact and externally confirmed physical row ends; FOV/RowMap anchors are not row ends.");

    module.def("assess_region_support", [](py::handle mask, py::handle box_xyxy,
                                          py::handle max_outside_fraction) {
        if (!py::isinstance<py::array>(mask))
            throw py::value_error("mask must be a NumPy bool array");
        const auto array = py::reinterpret_borrow<py::array>(mask);
        if (!array.dtype().is(py::dtype::of<bool>()))
            throw py::value_error("mask dtype must be bool; implicit conversion is not allowed");
        if (array.ndim() != 2 || array.shape(0) <= 0 || array.shape(1) <= 0)
            throw py::value_error("mask must have two nonempty dimensions");
        if (!(array.flags() & py::array::c_style))
            throw py::value_error("mask must be C-contiguous");
        const auto box = sequence(box_xyxy, "box_xyxy");
        if (box.size() != 4) throw py::value_error("box_xyxy must have length 4");
        std::array<std::int64_t, 4> coordinates{};
        for (std::size_t i = 0; i < coordinates.size(); ++i)
            coordinates[i] = integer(box[i], "box_xyxy coordinate");
        const auto assessment = fb::assess_region_support(static_cast<const bool*>(array.data()),
            static_cast<std::size_t>(array.shape(0)), static_cast<std::size_t>(array.shape(1)),
            coordinates, number(max_outside_fraction, "max_outside_fraction"));
        auto result = guarded();
        result["status"] = assessment.status;
        result["mask_area_pixels"] = assessment.mask_area_pixels;
        result["inside_pixels"] = assessment.inside_pixels;
        result["outside_pixels"] = assessment.outside_pixels;
        result["outside_fraction"] = assessment.outside_fraction;
        result["box_coverage_fraction"] = assessment.box_coverage_fraction;
        result["touches_image_edge"] = assessment.touches_image_edge;
        result["max_outside_fraction"] = assessment.max_outside_fraction;
        result["semantic_verified"] = false;
        result["ego_vehicle_verified"] = false;
        return result;
    }, py::arg("mask"), py::arg("box_xyxy"), py::arg("max_outside_fraction") = 0.15,
       "Count actual true pixels inside/outside a half-open original-image proposal [x0,y0,x1,y1]. "
       "Requires a 2D C-contiguous NumPy bool mask of at most 16777216 pixels; never converts, clips "
       "or modifies inputs. Read-only masks are accepted. Box coverage is inside pixels / box area, "
       "not IoU or semantic ground truth. Spatial consistency verifies neither semantics, ego-vehicle "
       "membership nor maneuver safety; authorization remains unknown.");

    module.def("project_depth", [](py::handle pixel_xy, py::handle depth_m, py::handle intrinsics,
                                  py::handle world_from_camera, py::handle image_size, py::handle max_range_m) {
        const auto size = sequence(image_size, "image_size");
        if (size.size() != 2) throw py::value_error("image_size must have length 2");
        std::array<int, 2> dimensions{};
        for (std::size_t i = 0; i < 2; ++i) {
            const auto dimension = integer(size[i], "image_size");
            if (dimension < 1 || dimension > std::numeric_limits<int>::max())
                throw py::value_error("image_size dimensions must be positive int32 values");
            dimensions[i] = static_cast<int>(dimension);
        }
        return fb::project_depth(numbers<2>(pixel_xy, "pixel_xy"), number(depth_m, "depth_m"),
            numbers<4>(intrinsics, "intrinsics"), numbers<16>(world_from_camera, "world_from_camera"),
            dimensions, number(max_range_m, "max_range_m"));
    }, py::arg("pixel_xy"), py::arg("depth_m"), py::arg("intrinsics"), py::arg("world_from_camera"),
       py::arg("image_size"), py::arg("max_range_m") = 30.0,
       "Only undistorted original-image pixels and optical-axis depth in meters. Camera: x right, y down, "
       "z forward; output world ENU via row-major rigid world_from_camera. No plane fallback. "
       "Invalid calibration raises ValueError; invalid sample or Euclidean range returns None.");

    py::class_<fb::EvidenceMap>(module, "EvidenceMap")
        .def(py::init([](py::handle map_id, py::handle epoch, py::handle cell_size_m,
                         py::handle ttl_s, py::handle min_views, py::handle min_baseline_m, py::handle max_cells) {
            const auto views = integer(min_views, "min_views");
            const auto cells = integer(max_cells, "max_cells");
            fb::require(views >= 1 && views <= static_cast<std::int64_t>(fb::EvidenceMap::max_groups_per_cell),
                        "min_views must be in [1, 256]");
            fb::require(cells >= 1 && cells <= static_cast<std::int64_t>(fb::EvidenceMap::max_configured_cells),
                        "max_cells must be in [1, 1000000]");
            return fb::EvidenceMap(text(map_id, "map_id"), text(epoch, "epoch"), number(cell_size_m, "cell_size_m"),
                number(ttl_s, "ttl_s"), static_cast<int>(views), number(min_baseline_m, "min_baseline_m"),
                static_cast<std::size_t>(cells));
        }), py::arg("map_id"), py::arg("epoch"), py::arg("cell_size_m") = 0.25,
            py::arg("ttl_s") = 5.0, py::arg("min_views") = 2, py::arg("min_baseline_m") = 0.5,
            py::arg("max_cells") = 200000,
            "Fixed epoch; no automatic reset/eviction. Retained group tombstones prevent replay refresh. "
            "All timestamps share one clock. Use a new map for an actual new epoch.")
        .def("update", [](fb::EvidenceMap& map, py::handle observation) {
            const auto update = map.update(observation_from_python(observation));
            return update_to_python(map, update);
        }, py::arg("observation"),
            "Atomic dict input: frame_id, evidence_group, capture_time, publish_time, epoch, points_xy_m, "
            "semantic, semantic_score, position_sigma_m, camera_xy_m, geometry_source='measured_depth'. "
            "IDs/labels are strings; times/XY/sigma are metric finite numbers. Same frame/prompts/tracks "
            "must share evidence_group. New groups require pairwise baseline and distinct frames to count. "
            "Returns accepted/reason and counts; malformed types raise ValueError. Duplicate labels preserve "
            "original sources, time, score, sigma and camera. Same-group conflicts share the original group TTL.")
        .def("snapshot", [](fb::EvidenceMap& map, py::handle now) {
            return snapshot_to_python(map, number(now, "now"));
        }, py::arg("now"),
            "Returns guarded metadata, status_counts, cells with sources and independent_views. "
            "TTL is inclusive at capture_time+ttl_s; unknown remains candidate, multiple labels conflict. "
            "Expired cells have semantic unknown; historical sources remain inactive. Sigma is a conservative "
            "cell lifetime maximum, not reduced by repeated evidence. now advances a monotonic watermark; "
            "backward/nonfinite snapshots raise ValueError. Consistent never authorizes maneuvering.");
}
