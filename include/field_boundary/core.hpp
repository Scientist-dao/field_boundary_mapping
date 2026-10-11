#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace field_boundary {

using XY = std::array<double, 2>;
using XYZ = std::array<double, 3>;
using Key = std::array<std::int64_t, 2>;

inline void require(bool condition, const std::string& reason) {
    if (!condition) throw std::invalid_argument(reason);
}

inline bool fraction(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

inline bool identifier(const std::string& value, std::size_t limit = 256) {
    return !value.empty() && value.size() <= limit &&
           value.find('\0') == std::string::npos &&
           value.find_first_not_of(" \t\r\n") != std::string::npos;
}

struct HeadlandAssessment {
    std::string hypothesis = "unknown";
    std::string status = "unknown";
    std::vector<std::string> reasons;
};

inline HeadlandAssessment assess_headland(double open_ground, double crop_contact,
                                         int row_ends = 0, bool rows_continue = false,
                                         bool fov_edge = false) {
    require(fraction(open_ground), "open_ground_fraction must be finite and in [0, 1]");
    require(fraction(crop_contact), "crop_contact_fraction must be finite and in [0, 1]");
    require(row_ends >= 0, "confirmed_row_ends must be nonnegative");
    HeadlandAssessment result;
    if (rows_continue) {
        result.hypothesis = "field_road_or_gap";
        result.status = "conflicting_headland";
        result.reasons = {"rows_continue"};
        return result;
    }
    if (open_ground < 0.01) result.reasons.push_back("insufficient_open_ground");
    if (crop_contact < 0.01) result.reasons.push_back("insufficient_crop_contact");
    if (!result.reasons.empty()) return result;
    if (fov_edge) result.reasons.push_back("row_ends_at_fov_edge");
    if (row_ends < 2) result.reasons.push_back("insufficient_confirmed_row_ends");
    if (!result.reasons.empty()) {
        result.hypothesis = "headland_or_field_road";
        result.status = "ambiguous";
    } else {
        result.hypothesis = "candidate_headland";
        result.status = "candidate";
        result.reasons = {"multiple_confirmed_row_ends_with_crop_contact"};
    }
    return result;
}

inline constexpr std::size_t max_region_support_pixels = 16777216;

struct RegionSupportAssessment {
    std::string status = "empty_mask";
    std::size_t mask_area_pixels = 0;
    std::size_t inside_pixels = 0;
    std::size_t outside_pixels = 0;
    std::optional<double> outside_fraction;
    double box_coverage_fraction = 0;
    bool touches_image_edge = false;
    double max_outside_fraction = 0.15;
};

// Read-only row-major mask; box_xyxy is a half-open proposal in original-image pixels.
// Spatial support is neither semantic verification, IoU nor maneuver authorization.
inline RegionSupportAssessment assess_region_support(
        const bool* mask, std::size_t height, std::size_t width,
        const std::array<std::int64_t, 4>& box_xyxy, double max_outside_fraction = 0.15) {
    require(height > 0 && width > 0, "mask dimensions must be nonempty");
    require(width <= max_region_support_pixels / height, "mask is limited to 16777216 pixels");
    require(mask != nullptr, "mask data must not be null");
    require(fraction(max_outside_fraction) && max_outside_fraction < 1.0,
            "max_outside_fraction must be finite and in [0, 1)");
    require(box_xyxy[0] >= 0 && box_xyxy[1] >= 0 &&
            box_xyxy[0] < box_xyxy[2] && box_xyxy[1] < box_xyxy[3] &&
            box_xyxy[2] <= static_cast<std::int64_t>(width) &&
            box_xyxy[3] <= static_cast<std::int64_t>(height),
            "box_xyxy must be nonempty and within the original image");
    const auto x0 = static_cast<std::size_t>(box_xyxy[0]);
    const auto y0 = static_cast<std::size_t>(box_xyxy[1]);
    const auto x1 = static_cast<std::size_t>(box_xyxy[2]);
    const auto y1 = static_cast<std::size_t>(box_xyxy[3]);
    RegionSupportAssessment result;
    result.max_outside_fraction = max_outside_fraction;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            // Inspect storage as a byte: a true mask value need not be encoded as 1.
            if (*reinterpret_cast<const unsigned char*>(mask + y * width + x) == 0) continue;
            ++result.mask_area_pixels;
            if (x >= x0 && x < x1 && y >= y0 && y < y1) ++result.inside_pixels;
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
                result.touches_image_edge = true;
        }
    }
    result.outside_pixels = result.mask_area_pixels - result.inside_pixels;
    result.box_coverage_fraction = static_cast<double>(result.inside_pixels) / ((x1 - x0) * (y1 - y0));
    if (result.mask_area_pixels == 0) return result;
    result.outside_fraction = static_cast<double>(result.outside_pixels) / result.mask_area_pixels;
    result.status = result.inside_pixels > 0 && *result.outside_fraction <= max_outside_fraction
        ? "spatially_consistent" : "outside_proposal";
    return result;
}

inline std::optional<XYZ> project_depth(const XY& pixel, double depth,
                                      const std::array<double, 4>& intrinsics,
                                      const std::array<double, 16>& transform,
                                      const std::array<int, 2>& image_size,
                                      double max_range = 30.0) {
    for (double value : intrinsics)
        require(std::isfinite(value), "intrinsics must be finite");
    require(intrinsics[0] > 0 && intrinsics[1] > 0, "fx and fy must be positive");
    require(image_size[0] > 0 && image_size[1] > 0, "image_size must be positive");
    require(std::isfinite(max_range) && max_range > 0, "max_range_m must be finite and positive");
    for (double value : transform)
        require(std::isfinite(value), "world_from_camera must be finite");
    for (int column = 0; column < 4; ++column)
        require(std::abs(transform[12 + column] - (column == 3 ? 1.0 : 0.0)) <= 1e-9,
                "world_from_camera last row must be [0, 0, 0, 1]");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0;
            for (int row = 0; row < 3; ++row)
                dot += transform[4 * row + i] * transform[4 * row + j];
            require(std::abs(dot - (i == j ? 1.0 : 0.0)) <= 1e-6,
                    "world_from_camera rotation must be orthonormal");
        }
    }
    const double determinant =
        transform[0] * (transform[5] * transform[10] - transform[6] * transform[9]) -
        transform[1] * (transform[4] * transform[10] - transform[6] * transform[8]) +
        transform[2] * (transform[4] * transform[9] - transform[5] * transform[8]);
    require(std::abs(determinant - 1.0) <= 1e-6,
            "world_from_camera rotation determinant must be +1");
    if (!std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
        pixel[0] < 0 || pixel[1] < 0 || pixel[0] >= image_size[0] || pixel[1] >= image_size[1] ||
        !std::isfinite(depth) || depth <= 0 || depth > max_range)
        return std::nullopt;
    const XYZ camera = {(pixel[0] - intrinsics[2]) / intrinsics[0] * depth,
                        (pixel[1] - intrinsics[3]) / intrinsics[1] * depth, depth};
    if (std::hypot(camera[0], camera[1], camera[2]) > max_range ||
        !std::isfinite(camera[0]) || !std::isfinite(camera[1]))
        return std::nullopt;
    XYZ world{};
    for (int row = 0; row < 3; ++row) {
        world[row] = transform[4 * row + 3];
        for (int column = 0; column < 3; ++column)
            world[row] += transform[4 * row + column] * camera[column];
        if (!std::isfinite(world[row])) return std::nullopt;
    }
    return world;
}

struct Observation {
    std::string frame_id;
    std::string evidence_group;
    double capture_time = 0;
    double publish_time = 0;
    std::string epoch;
    std::vector<XY> points;
    std::string semantic;
    double semantic_score = 0;
    double position_sigma_m = 0;
    XY camera{};
    std::string geometry_source;
};

struct Source {
    std::string frame_id;
    double capture_time;
    double publish_time;
    std::string semantic;
    double semantic_score;
    double position_sigma_m;
    XY camera;
};

struct Group {
    std::string id;
    double capture_time;
    XY camera;
    std::vector<Source> sources;
};

struct Cell {
    std::vector<Group> groups;
    double position_sigma_m = 0;
};

struct UpdateResult {
    bool accepted = false;
    std::string reason;
    std::size_t touched_cells = 0;
    std::size_t added_cells = 0;
    std::size_t added_sources = 0;
    std::size_t duplicate_cells = 0;
    std::size_t total_cells = 0;
    std::size_t total_sources = 0;
};

struct CellState {
    std::string status;
    std::string semantic;
    std::vector<std::string> labels;
    std::vector<std::size_t> counted_groups;
    std::size_t active_groups = 0;
};

class EvidenceMap {
public:
    static constexpr std::size_t max_points_per_observation = 100000;
    static constexpr std::size_t max_groups_per_cell = 256;
    static constexpr std::size_t max_labels_per_group = 16;
    static constexpr std::size_t max_total_sources = 1000000;
    static constexpr std::size_t max_configured_cells = 1000000;
    static constexpr double max_abs_coordinate_m = 1000000.0;
    static constexpr double max_abs_cell_index = 1000000000.0;

    EvidenceMap(std::string map_id, std::string epoch, double cell_size_m = 0.25,
                double ttl_s = 5.0, int min_views = 2, double min_baseline_m = 0.5,
                std::size_t max_cells = 200000)
        : map_id_(std::move(map_id)), epoch_(std::move(epoch)), cell_size_(cell_size_m),
          ttl_(ttl_s), min_views_(min_views), baseline_(min_baseline_m), max_cells_(max_cells) {
        require(identifier(map_id_), "map_id must be a nonempty string of at most 256 bytes");
        require(identifier(epoch_), "epoch must be a nonempty string of at most 256 bytes");
        require(std::isfinite(cell_size_) && cell_size_ > 0 && cell_size_ <= max_abs_coordinate_m,
                "cell_size_m must be finite and in (0, 1000000]");
        require(std::isfinite(ttl_) && ttl_ > 0, "ttl_s must be finite and positive");
        require(min_views_ >= 1 && min_views_ <= static_cast<int>(max_groups_per_cell),
                "min_views must be in [1, 256]");
        require(std::isfinite(baseline_) && baseline_ >= 0,
                "min_baseline_m must be finite and nonnegative");
        require(max_cells_ >= 1 && max_cells_ <= max_configured_cells,
                "max_cells must be in [1, 1000000]");
    }

    UpdateResult update(const Observation& observation) {
        const auto reject = [this](const std::string& reason) {
            UpdateResult result;
            result.reason = reason;
            result.total_cells = cells_.size();
            result.total_sources = source_count_;
            return result;
        };
        if (!identifier(observation.frame_id)) return reject("invalid_frame_id");
        if (!identifier(observation.evidence_group)) return reject("invalid_evidence_group");
        if (observation.epoch != epoch_) return reject("epoch_mismatch");
        if (observation.geometry_source != "measured_depth") return reject("geometry_not_measured_depth");
        if (!std::isfinite(observation.capture_time) || !std::isfinite(observation.publish_time))
            return reject("nonfinite_timestamp");
        if (observation.capture_time > observation.publish_time) return reject("capture_after_publish");
        if (last_capture_ && observation.capture_time < *last_capture_) return reject("out_of_order_capture");
        const double now = watermark_ ? std::max(*watermark_, observation.publish_time) : observation.publish_time;
        if (now - observation.capture_time > ttl_) return reject("capture_expired");
        if (!fraction(observation.semantic_score)) return reject("invalid_semantic_score");
        if (!std::isfinite(observation.position_sigma_m) || observation.position_sigma_m < 0)
            return reject("invalid_position_sigma_m");
        if (!identifier(observation.semantic, 128)) return reject("invalid_semantic");
        if (!valid_xy(observation.camera)) return reject("invalid_camera_xy_m");
        if (observation.points.empty()) return reject("empty_points");
        if (observation.points.size() > max_points_per_observation) return reject("too_many_points");
        std::set<Key> keys;
        for (const auto& point : observation.points) {
            if (!valid_xy(point)) return reject("invalid_point_xy_m");
            const double x = std::floor(point[0] / cell_size_);
            const double y = std::floor(point[1] / cell_size_);
            if (!std::isfinite(x) || !std::isfinite(y) ||
                std::abs(x) > max_abs_cell_index || std::abs(y) > max_abs_cell_index)
                return reject("cell_index_out_of_range");
            keys.insert({static_cast<std::int64_t>(x), static_cast<std::int64_t>(y)});
        }
        UpdateResult result;
        result.touched_cells = keys.size();
        for (const auto& key : keys)
            if (cells_.find(key) == cells_.end()) ++result.added_cells;
        if (result.added_cells > max_cells_ - cells_.size()) return reject("max_cells_exceeded");

        std::map<Key, Cell> staged;
        for (const auto& key : keys) {
            const auto existing = cells_.find(key);
            const Cell* cell = existing == cells_.end() ? nullptr : &existing->second;
            std::size_t group_index = cell ? cell->groups.size() : 0;
            if (cell) {
                for (std::size_t i = 0; i < cell->groups.size(); ++i) {
                    if (cell->groups[i].id == observation.evidence_group) {
                        group_index = i;
                        break;
                    }
                }
            }
            const bool new_group = !cell || group_index == cell->groups.size();
            if (!new_group) {
                const auto& group = cell->groups[group_index];
                if (std::any_of(group.sources.begin(), group.sources.end(), [&](const Source& source) {
                        return source.semantic == observation.semantic;
                    })) {
                    ++result.duplicate_cells;
                    continue;
                }
                if (group.sources.size() >= max_labels_per_group) return reject("max_labels_per_group_exceeded");
            } else if (cell && cell->groups.size() >= max_groups_per_cell) {
                return reject("max_groups_per_cell_exceeded");
            }
            if (source_count_ + result.added_sources >= max_total_sources)
                return reject("max_total_sources_exceeded");
            Cell patch = cell ? *cell : Cell{};
            if (new_group)
                patch.groups.push_back({observation.evidence_group, observation.capture_time, observation.camera, {}});
            patch.groups[group_index].sources.push_back({observation.frame_id, observation.capture_time,
                observation.publish_time, observation.semantic, observation.semantic_score,
                observation.position_sigma_m, observation.camera});
            patch.position_sigma_m = std::max(patch.position_sigma_m, observation.position_sigma_m);
            staged.emplace(key, std::move(patch));
            ++result.added_sources;
        }
        // All validation and allocation precede commit; map::merge transfers new nodes.
        for (auto it = staged.begin(); it != staged.end();) {
            const auto existing = cells_.find(it->first);
            if (existing == cells_.end()) {
                ++it;
            } else {
                existing->second.groups.swap(it->second.groups);
                existing->second.position_sigma_m = it->second.position_sigma_m;
                it = staged.erase(it);
            }
        }
        cells_.merge(staged);
        source_count_ += result.added_sources;
        last_capture_ = observation.capture_time;
        watermark_ = now;
        result.accepted = true;
        result.reason = result.added_sources ? "accepted" : "duplicate_evidence";
        result.total_cells = cells_.size();
        result.total_sources = source_count_;
        return result;
    }

    void advance_snapshot(double now) {
        require(std::isfinite(now), "snapshot now must be finite");
        require(!watermark_ || now >= *watermark_, "snapshot time precedes monotonic watermark");
        watermark_ = now;
    }

    bool active(const Group& group) const {
        return watermark_ && *watermark_ - group.capture_time <= ttl_;
    }

    CellState state(const Cell& cell) const {
        CellState result;
        std::set<std::string> labels;
        for (std::size_t index = 0; index < cell.groups.size(); ++index) {
            const auto& group = cell.groups[index];
            if (!active(group)) continue;
            ++result.active_groups;
            for (const auto& source : group.sources) labels.insert(source.semantic);
            bool independent = true;
            for (const auto previous : result.counted_groups) {
                const auto& counted = cell.groups[previous];
                if (std::hypot(group.camera[0] - counted.camera[0], group.camera[1] - counted.camera[1]) < baseline_)
                    independent = false;
                for (const auto& source : group.sources)
                    for (const auto& old_source : counted.sources)
                        if (source.frame_id == old_source.frame_id) independent = false;
            }
            if (independent) result.counted_groups.push_back(index);
        }
        result.labels.assign(labels.begin(), labels.end());
        result.semantic = labels.empty() ? "unknown" : labels.size() > 1 ? "conflict" : *labels.begin();
        if (!result.active_groups) result.status = "expired";
        else if (labels.size() > 1) result.status = "conflict";
        else if (result.semantic != "unknown" && result.counted_groups.size() >= static_cast<std::size_t>(min_views_))
            result.status = "consistent";
        else result.status = "candidate";
        return result;
    }

    const std::string& map_id() const { return map_id_; }
    const std::string& epoch() const { return epoch_; }
    double cell_size() const { return cell_size_; }
    double ttl() const { return ttl_; }
    int min_views() const { return min_views_; }
    double baseline() const { return baseline_; }
    std::size_t max_cells() const { return max_cells_; }
    std::size_t source_count() const { return source_count_; }
    const std::optional<double>& watermark() const { return watermark_; }
    const std::optional<double>& last_capture() const { return last_capture_; }
    const std::map<Key, Cell>& cells() const { return cells_; }

private:
    static bool valid_xy(const XY& xy) {
        return std::isfinite(xy[0]) && std::isfinite(xy[1]) &&
               std::abs(xy[0]) <= max_abs_coordinate_m && std::abs(xy[1]) <= max_abs_coordinate_m;
    }

    std::string map_id_;
    std::string epoch_;
    double cell_size_;
    double ttl_;
    int min_views_;
    double baseline_;
    std::size_t max_cells_;
    std::size_t source_count_ = 0;
    std::optional<double> watermark_;
    std::optional<double> last_capture_;
    std::map<Key, Cell> cells_;
};

}  // namespace field_boundary
