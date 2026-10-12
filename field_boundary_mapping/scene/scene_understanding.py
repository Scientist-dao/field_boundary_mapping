import json
import math

from field_boundary_mapping.proposals.run_proposals import REGION_PROMPTS, fields, identifier


PROMPT_VERSION = "functional-scene-v1"
SCHEMA_VERSION = "field_functional_scene_v1"
ROLES = ("current_field", "neighbor_field", "headland_candidate", "farm_road", "obstacle", "unknown")
SURFACES = (*REGION_PROMPTS, "mixed", "unknown")
RELATIONS = ("headland_of", "access_road_to", "adjacent_to", "separated_from", "obstacle_in")
BOUNDARY_TYPES = ("field_headland", "field_road", "between_fields", "obstacle_edge", "unknown")
GATES = {"semantic_verified": False, "physical_field_boundary_verified": False,
         "maneuver_verified": False, "annotation_used_for_inference": False}

INSTRUCTION_V1 = '''Understand the agricultural scene in the LAST image, not merely its surface classes.
You may receive earlier images of the same camera in chronological order. Use them only as past context; all output locations refer to the LAST image. Earlier presence does not prove a currently occluded boundary. No future images, maps, annotations, navigation commands or known operation stage are supplied.
Identify visible functional regions: current_field (the field the camera vehicle appears to be working in), neighbor_field (a distinct neighboring field), headland_candidate (an end-of-row working strip associated with a field), farm_road (a transit/access corridor), obstacle, unknown. The visible vehicle hood is an obstacle candidate too, never erase it. A bare patch is NOT automatically a headland. Distinguish farm roads from headlands by row ends, orientation, access layout and temporal visual context. If the active field cannot be identified, use unknown and explain the ambiguity. A headland is a functional hypothesis, NOT a safely turnable area.
Keep role separate from visible surface. Allowed surface values: crop, harvested_ground, ground, water, ditch, vehicle, person, tarp, woody_vegetation, pole, mixed, unknown. Split different functional regions even when they share a surface; do not invent invisible far boundaries. Output up to 8 regions. Use a coarse simple polygon to mark ONLY each visible region in the LAST image; do not use a whole-image box for a narrow strip. Polygon vertices are integer [x,y] pairs in 0..1000 normalized image coordinates, at least 3 and at most 24 distinct vertices, no repeated closing point. All vertices must lie inside the image. A polygon is only approximate localization, not a physical property boundary. Mixed/unknown surfaces may remain unrefined by the segmentation stage.
For each role state concise visual evidence and specific uncertainty. When spatial extent is not supportable, omit that region and explain it in unknowns. Do not force current_field or headland_candidate into every image. Multiple polygons may overlap because of occlusion or ambiguity; do not hide this ambiguity.
Output up to 16 directed relations between region IDs, using headland_of (headland -> field), access_road_to (road -> field), adjacent_to, separated_from, obstacle_in (obstacle -> containing region). Relations are hypotheses, not verified adjacency.
Output up to 12 OPEN polylines for directly visible functional boundary segments only. Each has 2..32 integer [x,y] points in the same coordinates, side_a/side_b region IDs or the literal unknown, kind from field_headland, field_road, between_fields, obstacle_edge, unknown, and visual evidence. Do not close invisible boundaries or join segments across occlusion. If the boundary is unobservable return no line and report why. Do NOT derive these lines merely from surface-class adjacency.
Never claim verified semantics, legal ownership, metric distances, traversability or safe turning. Do not produce commands or code. Return ONE JSON object only with exactly these four top-level keys:
{"regions":[{"region_id":"r1","role":"unknown","surface":"ground","polygon":[[10,600],[500,600],[500,900],[10,900]],"evidence":"visible bare patch; function unclear","uncertainties":["road versus headland unclear"]}],"relations":[],"boundaries":[],"unknowns":["physical field extent and turning clearance are unknown"]}
Relation format: {"subject":"r1","relation":"adjacent_to","object":"r2","evidence":"visible spatial arrangement"}.
Boundary format: {"boundary_id":"b1","side_a":"r1","side_b":"r2","kind":"unknown","points":[[10,600],[500,600]],"evidence":"visible separating edge"}.
An empty regions/relations/boundaries list is valid when evidence is insufficient; explain in unknowns.'''

INSTRUCTION_V2 = '''Describe visible agricultural functional regions in the LAST (CURRENT) image using visual evidence, not an assumed working stage.
Earlier images are PAST context only. Re-localize every region and boundary in CURRENT: camera rotation changes coordinates. Past visibility does not prove current visibility. No maps, annotations, future frames or operation commands are available.

Separate what is visible (surface) from its inferred function (role):
- current_field: a field supported as the vehicle's active working field by visible working-row alignment or a visually traceable working path. Merely facing a field, driving beside it, tire tracks, or bare soil is insufficient. When active-field identity is unclear, use unknown with the visible surface; do not force a current_field.
- neighbor_field: a visibly distinct field, not a tree line, windbreak, horizon band, or duplicated portion of the same field. If the relation to an active field is unclear, use unknown. Never invent a second field just to express uncertainty about one field.
- headland_candidate: requires BOTH visible row termination and a working strip across those row ends. Describe those two cues in evidence. A road junction, bare patch, distant horizon strip, or field edge alone is insufficient: use farm_road or unknown instead. Omit an unsupported headland hypothesis and state what is missing in unknowns.
- farm_road: a visually supported transit/access corridor; tire tracks, corridor continuity, connections and parked vehicles can support this role. Do not assign the same patch as headland merely because it adjoins crops.
- obstacle: visible people, external vehicles, poles and other obstacles; include the camera vehicle hood/tires too. Scan the whole image for small external vehicles and people, not only the foreground hood. Give each distinguishable object its own localized region and surface. Within the 8-region budget, prioritize visible people and nearby external vehicles over speculative field subdivisions. If an object is too small to localize, mention it in unknowns instead of claiming absence.
- unknown: visible region whose function is not supportable. Trees may have surface woody_vegetation without being a neighboring field. Harvested_ground requires visible stubble, cut rows or crop residue, not just dirt or tracks. Shadows are lighting, not a new ground surface.

Allowed surface values: crop, harvested_ground, ground, water, ditch, vehicle, person, tarp, woody_vegetation, pole, mixed, unknown.
Use up to 8 regions. Follow visible sloping edges with a coarse simple polygon; do not replace a diagonal region by a horizontal rectangle or truncate a continuous field at an arbitrary interior coordinate. Mark only CURRENT visible extent, excluding sky and unrelated areas as far as the coarse polygon permits. Polygons localize hypotheses, not physical property boundaries. Overlap due to occlusion or genuinely uncertain functions is allowed; explain it, do not fabricate disjoint regions. Mixed/unknown surfaces may stay unrefined.
A polygon is a list of 3..24 DISTINCT integer [x,y] pairs, each coordinate in 0..1000 normalized to CURRENT. No repeated closing point, no self-intersections. A box such as [[x0,y0,x1,y1]] is INVALID; each vertex MUST contain exactly TWO integers. For each region provide concise evidence tied to visible cues and specific uncertainties.

Use up to 16 relations: headland_of (headland -> field), access_road_to (road -> field), adjacent_to, separated_from, obstacle_in (obstacle -> containing region). Do not force a road-to-field relation when the field role is unknown. Relations are hypotheses, not verified topology or ownership.
Use up to 12 OPEN boundary polylines for directly visible functional edges only. Each has 2..32 distinct integer [x,y] points in the same coordinates. Trace actual sloping/curved separating edges, never a polygon/box top edge. A vehicle silhouette is curved; its hood top is not a whole-image horizontal line. Do not bridge occlusions or close invisible borders. A tree-line base is not automatically a between_fields boundary. A surface-mask contour does not establish a functional boundary. If function or visible location is unsupported, omit the line and explain in unknowns. Allowed kinds: field_headland, field_road, between_fields, obstacle_edge, unknown. Each side is a region ID or the literal unknown.

Never claim verified semantics, legal ownership, metric distances, traversability or safe turning. A headland hypothesis is NOT a safely turnable area. Do not output commands or code. Return ONE JSON object only with exactly regions, relations, boundaries, unknowns. Empty lists are valid when evidence is insufficient; explain in unknowns. Use these formats (example coordinates are NOT a template for the current image):
{"regions":[{"region_id":"r1","role":"unknown","surface":"ground","polygon":[[20,610],[480,530],[620,820],[30,910]],"evidence":"visible soil; function unclear","uncertainties":["row termination not visible"]}],"relations":[],"boundaries":[],"unknowns":["active field identity and turning clearance unknown"]}
Relation: {"subject":"r1","relation":"adjacent_to","object":"r2","evidence":"visible spatial arrangement"}.
Boundary: {"boundary_id":"b1","side_a":"r1","side_b":"r2","kind":"unknown","points":[[20,610],[480,530]],"evidence":"visible separating edge"}.
Before returning, check CURRENT coordinates, two integers per vertex, visible evidence for each field/headland role, separate external obstacles, and no invented box-edge boundaries.'''

INSTRUCTION_V3 = '''Analyze agricultural functions in the LAST image (CURRENT). Earlier images are PAST context only; camera motion changes image coordinates. Locate everything again in CURRENT. Use only visible evidence, not an assumed operation stage, future images, maps or annotations.

First inspect visible people, external vehicles, the camera vehicle, crops, soil and roads. Give distinguishable nearby people and external vehicles separate localized regions; retain the camera hood/tires too. Prioritize them over speculative field subdivisions. Mention objects too indistinct to localize in unknowns; do not claim absence.
Then assign role separately from surface:
- current_field: visible working rows or a traceable working path place the camera vehicle within the active crop-working area. Driving on a road beside crops does NOT identify an active field, even parallel to rows. If identity is unclear, keep the crop surface with role unknown.
- neighbor_field: a visibly distinct field related to an identified working field; never a treeline or a duplicate portion of the same field.
- headland_candidate: BOTH visible row termination AND a working strip across those row ends, described in evidence. A road, junction, bare patch or horizon band alone is insufficient. Do not force a headland.
- farm_road: a visible transit/access corridor supported by layout, tracks or vehicles; it is not automatically a headland.
- obstacle: person, vehicle, pole or another visible obstacle.
- unknown: function lacks support; a known surface may still be given. Harvested_ground requires visible stubble, cut rows or residue, not merely soil. Shadows are not separate surfaces or functional regions.
Allowed surfaces: crop, harvested_ground, ground, water, ditch, vehicle, person, tarp, woody_vegetation, pole, mixed, unknown.

Localize each region with a simple polygon along its visible outline. Follow sloping edges; do not truncate continuous crops at an arbitrary interior coordinate. Occlusion-related overlap is allowed. Use 3..24 DISTINCT vertices ordered clockwise around the outline, without crossing edges or repeating the first vertex. Every vertex is an array of exactly TWO INTEGERS [x,y], normalized to CURRENT in 0..1000. A polygon encloses area, including for a thin pole. If localization is unsupported, omit the region and explain in unknowns.
Boundaries are optional OPEN polylines along directly visible separating edges, not copied polygon edges. Trace actual slopes/curves; a hood silhouette is not a full-width horizontal line. Never bridge occlusions. If an edge or its function is unclear, omit the boundary and explain. Neither polygons nor lines prove physical or legal field boundaries, metric clearance or safe turning. No commands or code.

Return ONE JSON object with exactly four keys: regions, relations, boundaries, unknowns. Each is an array; limits are 8, 16, 12, 16 entries respectively. Do not fill a budget with unsupported objects. All explanatory strings must be short.
EVERY region MUST contain exactly these SIX keys, including uncertainties even when it is []:
{"region_id":"r1","role":"unknown","surface":"ground","polygon":[[20,610],[480,530],[620,820],[30,910]],"evidence":"visible soil; function unclear","uncertainties":["row ends not visible"]}
The numbers above illustrate syntax only, not image locations. ALWAYS write polygon as nested [x,y] vertex arrays.
EVERY relation MUST contain exactly: {"subject":"r1","relation":"adjacent_to","object":"r2","evidence":"visible arrangement"}. Use existing distinct region IDs. Allowed types: adjacent_to, separated_from, headland_of (headland_candidate -> current_field or neighbor_field), access_road_to (farm_road -> current_field or neighbor_field), obstacle_in (obstacle -> containing region). Omit unsupported relations.
EVERY boundary MUST contain exactly: {"boundary_id":"b1","side_a":"r1","side_b":"r2","kind":"unknown","points":[[20,610],[480,530]],"evidence":"visible separating edge"}. Sides must be distinct existing region IDs or the literal unknown. Allowed kinds: field_headland, field_road, between_fields, obstacle_edge, unknown. Points are 2..32 DISTINCT integer [x,y] arrays; the line stays open.
Unknowns and every region's uncertainties are arrays of strings (uncertainties: at most 8). Never omit uncertainties. Empty region/relation/boundary arrays are valid with an explanation in unknowns. Before emitting JSON, check all six region keys, every vertex pair, CURRENT locations and visible evidence for each role.'''

INSTRUCTION_V4 = INSTRUCTION_V1 + '''
Important distinctions for this task:
Visible surface and functional certainty are independent. Keep clearly visible crops, soil, roads and the camera vehicle as localized regions even when their function is uncertain. Use role unknown with the visible surface and explain the uncertainty. Lack of evidence for current_field or headland_candidate is NOT a reason to return an empty scene or omit the camera hood/tires. Return an empty scene only when no visible region can be localized.
Scan the whole CURRENT image for external vehicles and people; localize distinguishable nearby objects separately, not only the camera vehicle. Small size alone does not mean absence. Re-localize all objects in CURRENT after camera movement; never copy past-frame coordinates.
A vehicle driving beside crops on a road does not establish current_field. A treeline is not neighbor_field. A headland_candidate needs BOTH visible row termination and a working strip across those ends. When those functional cues are missing, retain the visible region as unknown or farm_road rather than inventing a field or headland.
Follow the visible sloping outline of a region; do not cut continuous crops at an arbitrary internal coordinate. Trace visible boundary slopes/curves, not full-width horizontal substitutes for a hood silhouette. If boundary location is unclear, omit the line but keep supported regions.
Every region must contain all SIX keys: region_id, role, surface, polygon, evidence, uncertainties. Always include uncertainties, even when []. Write each polygon as 3..24 nested [x,y] integer vertex arrays, in clockwise outline order, enclosing nonzero area without crossings or a repeated closing point. Keep the exact JSON structure illustrated above.'''

INSTRUCTION_V5 = '''Understand the agricultural scene in the LAST image, not merely its surface classes.
You may receive earlier images of the same camera in chronological order. Use them only as past context; all output locations refer to the LAST image. Earlier presence does not prove a currently occluded boundary. No future images, maps, annotations, navigation commands or known operation stage are supplied.
Identify visible functional regions: current_field (the field the camera vehicle appears to be working in), neighbor_field (a distinct neighboring field), headland_candidate (an end-of-row working strip associated with a field), farm_road (a transit/access corridor), obstacle, unknown. The visible vehicle hood is an obstacle candidate too, never erase it. A bare patch is NOT automatically a headland. Distinguish farm roads from headlands by row ends, orientation, access layout and temporal visual context. If the active field cannot be identified, use unknown and explain the ambiguity. A headland is a functional hypothesis, NOT a safely turnable area.
Keep role separate from visible surface. Allowed surface values: crop, harvested_ground, ground, water, ditch, vehicle, person, tarp, woody_vegetation, pole, mixed, unknown. Split different functional regions even when they share a surface; do not invent invisible far boundaries. Output up to 8 regions.
Localize each visible region with ONE axis-aligned box: "box":[x1,y1,x2,y2] - exactly four integers in 0..1000 normalized image coordinates, x1 < x2 and y1 < y2, covering only that region's visible extent in the LAST image. Do not use a whole-image box for a narrow strip. Never write polygon vertices; the box is approximate localization only, and a segmentation model resolves the pixel outline afterwards. Mixed/unknown surfaces may remain unrefined by the segmentation stage.
For each role state concise visual evidence and specific uncertainty. When spatial extent is not supportable, omit that region and explain it in unknowns. Do not force current_field or headland_candidate into every image. Multiple boxes may overlap because of occlusion or ambiguity; do not hide this ambiguity.
Output up to 16 directed relations between region IDs, using headland_of (headland -> field), access_road_to (road -> field), adjacent_to, separated_from, obstacle_in (obstacle -> containing region). Relations are hypotheses, not verified adjacency.
Output up to 12 OPEN boundary segments for directly visible functional boundaries only. Each is a straight segment of EXACTLY two integer [x,y] points in the same 0..1000 coordinates, side_a/side_b region IDs or the literal unknown, kind from field_headland, field_road, between_fields, obstacle_edge, unknown, and visual evidence. Do not close invisible boundaries or join segments across occlusion. If the boundary is unobservable return no line and report why. Do NOT derive these lines merely from surface-class adjacency.
Never claim verified semantics, legal ownership, metric distances, traversability or safe turning. Do not produce commands or code. Return ONE JSON object only with exactly these four top-level keys:
{"regions":[{"region_id":"r1","role":"unknown","surface":"ground","box":[10,600,500,900],"evidence":"visible bare patch; function unclear","uncertainties":["road versus headland unclear"]}],"relations":[],"boundaries":[],"unknowns":["physical field extent and turning clearance are unknown"]}
Relation format: {"subject":"r1","relation":"adjacent_to","object":"r2","evidence":"visible spatial arrangement"}.
Boundary format: {"boundary_id":"b1","side_a":"r1","side_b":"r2","kind":"unknown","points":[[10,600],[500,600]],"evidence":"visible separating edge"}.
An empty regions/relations/boundaries list is valid when evidence is insufficient; explain in unknowns.'''

INSTRUCTION_V6 = '''Understand the agricultural scene in the LAST image, not merely its surface classes.
You may receive earlier images of the same camera in chronological order. Use them only as past context; all output locations refer to the LAST image. Earlier presence does not prove a currently occluded boundary. No future images, maps, annotations, navigation commands or known operation stage are supplied.
Identify visible functional regions: current_field (the field the camera vehicle appears to be working in), neighbor_field (a distinct neighboring field), headland_candidate (an end-of-row working strip associated with a field), farm_road (a transit/access corridor), obstacle, unknown. The visible vehicle hood is an obstacle candidate too, never erase it. A bare patch is NOT automatically a headland. Distinguish farm roads from headlands by row ends, orientation, access layout and temporal visual context. If the active field cannot be identified, use unknown and explain the ambiguity. A headland is a functional hypothesis, NOT a safely turnable area.
Keep role separate from visible surface. Allowed surface values: crop, harvested_ground, ground, water, ditch, vehicle, person, tarp, woody_vegetation, pole, mixed, unknown. Split different functional regions even when they share a surface; do not invent invisible far boundaries. Output up to 8 regions.
Localize each visible region with ONE axis-aligned box: "box":[x1,y1,x2,y2] - exactly four integers in 0..1000 normalized image coordinates, x1 < x2 and y1 < y2, covering only that region's visible extent in the LAST image. Do not use a whole-image box for a narrow strip. Never write polygon vertices; the box is approximate localization only, and a segmentation model resolves the pixel outline afterwards. Mixed/unknown surfaces may remain unrefined by the segmentation stage.
For each role state concise visual evidence and specific uncertainty. When spatial extent is not supportable, omit that region and explain it in unknowns. Do not force current_field or headland_candidate into every image. Multiple boxes may overlap because of occlusion or ambiguity; do not hide this ambiguity.
Output up to 16 directed relations between region IDs, using headland_of (headland -> field), access_road_to (road -> field), adjacent_to, separated_from, obstacle_in (obstacle -> containing region). Relations are hypotheses, not verified adjacency.
Output up to 12 OPEN boundary segments for directly visible functional boundaries only. Localize each segment with "segment":[x1,y1,x2,y2] - exactly four integers in the same 0..1000 coordinates, the straight segment from (x1,y1) to (x2,y2). Do not write polygon vertex arrays and do not write nested point pairs for a segment. Each boundary also needs side_a/side_b region IDs or the literal unknown, kind from field_headland, field_road, between_fields, obstacle_edge, unknown, and visual evidence. Do not close invisible boundaries or join segments across occlusion. If a boundary is unobservable, or you cannot state it as four integers, return an empty boundaries list and report why in unknowns. Do NOT derive lines merely from surface-class adjacency.
Keep every evidence, uncertainty and unknown string between 1 and 700 characters, splitting long explanations into several short strings.
Never claim verified semantics, legal ownership, metric distances, traversability or safe turning. Do not produce commands or code. Return ONE JSON object only with exactly these four top-level keys:
{"regions":[{"region_id":"r1","role":"unknown","surface":"ground","box":[10,600,500,900],"evidence":"visible bare patch; function unclear","uncertainties":["road versus headland unclear"]}],"relations":[],"boundaries":[],"unknowns":["physical field extent and turning clearance are unknown"]}
Relation format: {"subject":"r1","relation":"adjacent_to","object":"r2","evidence":"visible spatial arrangement"}.
Boundary format: {"boundary_id":"b1","side_a":"r1","side_b":"r2","kind":"unknown","segment":[10,600,500,600],"evidence":"visible separating edge"}.
An empty regions/relations/boundaries list is valid when evidence is insufficient; explain in unknowns.'''

INSTRUCTIONS = {"functional-scene-v1": INSTRUCTION_V1, "functional-scene-v2": INSTRUCTION_V2,
                "functional-scene-v3": INSTRUCTION_V3, "functional-scene-v4": INSTRUCTION_V4,
                "functional-scene-v5-box": INSTRUCTION_V5, "functional-scene-v6-box": INSTRUCTION_V6}


def _text(value, name, maximum=700):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} requires 1 to {maximum} characters")


def _list(value, limit, name):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} requires a list of at most {limit} entries")


def _enum(value, allowed, name):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Unsupported {name}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _segment(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(v) is not int or not 0 <= v <= 1000 for v in value)):
        raise ValueError("segment requires exactly four integers in 0..1000")
    if value[:2] == value[2:]:
        raise ValueError("segment endpoints must differ")


def _points(value, minimum, maximum, name):
    _list(value, maximum, name)
    if len(value) < minimum:
        raise ValueError(f"Too few {name}")
    for point in value:
        if (not isinstance(point, list) or len(point) != 2
                or any(type(v) is not int or not 0 <= v <= 1000 for v in point)):
            raise ValueError("Coordinates must be integer pairs in 0..1000")
    if len({tuple(point) for point in value}) != len(value):
        raise ValueError("Repeated point")


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    signs = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    return (signs[0] * signs[1] <= 0 and signs[2] * signs[3] <= 0
            and max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
            and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1])))


def _box(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(v) is not int or not 0 <= v <= 1000 for v in value)):
        raise ValueError("box requires exactly four integers in 0..1000")
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box requires x1 < x2 and y1 < y2")


def _polygon(points):
    _points(points, 3, 24, "polygon")
    edges = list(zip(points, points[1:] + points[:1]))
    if abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)) < 2:
        raise ValueError("Degenerate polygon")
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _intersects(a, b, c, d):
                raise ValueError("Self-intersecting polygon")


def parse_scene(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > 131072:
        raise ValueError("Scene response must be text of at most 128 KiB")
    scene = json.loads(text, object_pairs_hook=_unique_object)
    fields(scene, ("regions", "relations", "boundaries", "unknowns"), "scene")
    _list(scene["regions"], 8, "regions")
    _list(scene["relations"], 16, "relations")
    _list(scene["boundaries"], 12, "boundaries")
    _list(scene["unknowns"], 16, "unknowns")
    for value in scene["unknowns"]:
        _text(value, "unknown")
    if not scene["regions"] and not scene["unknowns"]:
        raise ValueError("Empty scene must explain missing evidence")
    by_id, polygons = {}, set()
    for region in scene["regions"]:
        has_box = isinstance(region, dict) and "box" in region
        has_polygon = isinstance(region, dict) and "polygon" in region
        if has_box and has_polygon:
            raise ValueError("A region localizes with polygon or box, not both")
        geometry_key = "box" if has_box else "polygon"
        fields(region, ("region_id", "role", "surface", geometry_key, "evidence", "uncertainties"), "region")
        identifier(region["region_id"], "region_id")
        if region["region_id"] in by_id or region["region_id"] == "unknown":
            raise ValueError("Duplicate or reserved region ID")
        by_id[region["region_id"]] = region
        _enum(region["role"], ROLES, "role")
        _enum(region["surface"], SURFACES, "surface")
        if has_box:
            _box(region["box"])
            key = (region["role"], region["surface"], tuple(region["box"]))
        else:
            _polygon(region["polygon"])
            key = (region["role"], region["surface"], tuple(sorted(map(tuple, region["polygon"]))))
        if key in polygons:
            raise ValueError("Duplicate functional region")
        polygons.add(key)
        _text(region["evidence"], "evidence")
        _list(region["uncertainties"], 8, "uncertainties")
        for value in region["uncertainties"]:
            _text(value, "uncertainty")
    relations = set()
    for relation in scene["relations"]:
        fields(relation, ("subject", "relation", "object", "evidence"), "relation")
        _enum(relation["subject"], by_id, "relation subject")
        _enum(relation["object"], by_id, "relation object")
        _enum(relation["relation"], RELATIONS, "relation type")
        key = (relation["subject"], relation["relation"], relation["object"])
        if key in relations or key[0] == key[2]:
            raise ValueError("Duplicate or self relation")
        relations.add(key)
        subject, target = by_id[key[0]]["role"], by_id[key[2]]["role"]
        if key[1] == "headland_of" and (subject != "headland_candidate" or target not in ("current_field", "neighbor_field")):
            raise ValueError("headland_of requires headland -> field")
        if key[1] == "access_road_to" and (subject != "farm_road" or target not in ("current_field", "neighbor_field")):
            raise ValueError("access_road_to requires road -> field")
        if key[1] == "obstacle_in" and subject != "obstacle":
            raise ValueError("obstacle_in requires an obstacle")
        _text(relation["evidence"], "relation evidence")
    boundary_ids = set()
    for boundary in scene["boundaries"]:
        has_segment = isinstance(boundary, dict) and "segment" in boundary
        if has_segment and isinstance(boundary, dict) and "points" in boundary:
            raise ValueError("A boundary localizes with points or segment, not both")
        fields(boundary, ("boundary_id", "side_a", "side_b", "kind", "segment" if has_segment else "points",
                          "evidence"), "boundary")
        identifier(boundary["boundary_id"], "boundary ID")
        if boundary["boundary_id"] in boundary_ids:
            raise ValueError("Duplicate boundary ID")
        boundary_ids.add(boundary["boundary_id"])
        for key in ("side_a", "side_b"):
            _enum(boundary[key], (*by_id, "unknown"), key)
        if boundary["side_a"] == boundary["side_b"]:
            raise ValueError("A boundary requires distinct sides")
        _enum(boundary["kind"], BOUNDARY_TYPES, "boundary kind")
        if has_segment:
            # Downstream geometry and rendering consume endpoints, so the declared four-integer
            # segment becomes the canonical two-endpoint form here, without changing its numbers.
            _segment(boundary["segment"])
            x1, y1, x2, y2 = boundary.pop("segment")
            boundary["points"] = [[x1, y1], [x2, y2]]
        _points(boundary["points"], 2, 32, "boundary points")
        _text(boundary["evidence"], "boundary evidence")
    return scene


def pixel_points(points, width, height):
    return [[min(width - 1, round(x * width / 1000)), min(height - 1, round(y * height / 1000))]
            for x, y in points]


def region_geometry_kind(region):
    return "vlm_box" if "box" in region else "vlm_polygon"


def full_width_box(region):
    """Audit predicate only: a box spanning the whole 0..1000 width localizes
    no horizontal extent. It never changes parsing success, role or surface."""
    box = region.get("box", ()) if isinstance(region, dict) else ()
    return len(box) == 4 and box[0] == 0 and box[2] == 1000


def region_corners(region):
    """Region outline as 0..1000 points, clockwise in image coordinates (y grows downward)."""
    if "box" in region:
        x1, y1, x2, y2 = region["box"]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return region["polygon"]


def region_outline(region, width, height):
    return pixel_points(region_corners(region), width, height)


def scene_queries(scene, width, height):
    queries = []
    for region in scene["regions"]:
        if region["surface"] not in REGION_PROMPTS:
            continue
        xs, ys = zip(*region_corners(region))
        box = [math.floor(min(xs) * width / 1000), math.floor(min(ys) * height / 1000),
               math.ceil(max(xs) * width / 1000), math.ceil(max(ys) * height / 1000)]
        queries.append({"proposal_id": region["region_id"], "semantic": region["surface"],
                        "box_xyxy": box, "prompt": REGION_PROMPTS[region["surface"]],
                        "relation": "unspecified", "rationale": region["evidence"]})
    return queries
