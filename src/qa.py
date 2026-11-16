"""Spoken Q&A engine for VisionQuest with Natural Language Distance & Colour.

Answers user questions based strictly on the latest camera frame tracks and pixels.
Implements:
1. Natural language distance categories:
   - VERY CLOSE (<= 1.0m or visual scale >= 65%)
   - CLOSE      (1.0m < d <= 2.0m or visual scale >= 40%)
   - MEDIUM     (2.0m < d <= 3.5m)
   - FAR        (> 3.5m)
   - UNCERTAIN  (Truncated or uncalibrated)
2. Direction in every object announcement:
   - "on your left"
   - "in the centre"
   - "on your right"
3. Distance comparison, nearest-object search, and approach/motion analysis.
4. Combined Object + Colour + Direction + Distance responses.
5. System control commands (repeat, silence, resume, help, unknown object).
"""

from typing import List, Optional
import numpy as np

from .contracts import TrackInfo
from .intent import Intent
from .color import (
    ColorResult,
    extract_object_color,
    get_scene_dominant_color,
    find_objects_with_color,
)
from .distance_categories import get_distance_info, DistanceInfo, DIRECTION_PHRASES


_LAST_SPOKEN_ANSWER: Optional[str] = None
_IS_SILENCED: bool = False


def is_silenced() -> bool:
    return _IS_SILENCED


def set_silenced(val: bool):
    global _IS_SILENCED
    _IS_SILENCED = val


def get_last_spoken() -> Optional[str]:
    return _LAST_SPOKEN_ANSWER


def _format_dir(direction: str) -> str:
    dir_key = direction.lower() if direction else "ahead"
    return DIRECTION_PHRASES.get(dir_key, "in the centre")


# ---------------------------------------------------------------------------
# Public Answer Entrypoint
# ---------------------------------------------------------------------------
def answer(
    intent: Intent,
    infos: List[TrackInfo],
    yolo_names: List[str],
    frame_bgr: Optional[np.ndarray] = None
) -> str:
    """Generate a concise spoken response for the given intent, tracks, and frame."""
    global _LAST_SPOKEN_ANSWER, _IS_SILENCED

    resp = _build_response(intent, infos, yolo_names, frame_bgr)

    # Don't overwrite cached answer if this was just a repeat request
    if intent.kind != "repeat":
        _LAST_SPOKEN_ANSWER = resp

    return resp


def _build_response(
    intent: Intent,
    infos: List[TrackInfo],
    yolo_names: List[str],
    frame_bgr: Optional[np.ndarray]
) -> str:
    global _IS_SILENCED

    # 1. System Control Commands
    if intent.kind == "repeat":
        if _LAST_SPOKEN_ANSWER:
            return _LAST_SPOKEN_ANSWER
        return "I haven't said anything yet."

    if intent.kind == "silence":
        _IS_SILENCED = True
        return "Audio alerts silenced. Say 'start listening again' or press V to resume."

    if intent.kind == "resume":
        _IS_SILENCED = False
        return "Audio alerts resumed."

    if intent.kind == "help":
        return (
            "You can ask where objects are, ask their distance such as 'how far is my bottle', "
            "check who is nearest, ask colours, or say 'be quiet' to silence alerts."
        )

    if intent.kind == "identify_object":
        return _identify_unnamed_object(infos, frame_bgr)

    # 2. Distance & Proximity Intents
    if intent.kind == "is_approaching":
        return _answer_is_approaching(intent, infos)

    if intent.kind == "compare_distance":
        return _answer_compare_distance(intent, infos)

    if intent.kind == "nearest_object":
        return _answer_nearest_object(intent, infos)

    if intent.kind == "is_anything_close":
        return _answer_is_anything_close(intent, infos)

    if intent.kind == "is_distance_verify":
        return _answer_distance_verify(intent, infos, frame_bgr)

    if intent.kind == "how_far":
        return _answer_how_far(intent, infos, frame_bgr)

    # 3. Colour Recognition Intents
    if intent.kind == "scene_color":
        return _answer_scene_color(frame_bgr)

    if intent.kind == "find_by_color":
        return _answer_find_by_color(intent, infos, frame_bgr)

    if intent.kind == "color_compare":
        return _answer_color_compare(intent, infos, frame_bgr)

    if intent.kind == "color_choice":
        return _answer_color_choice(intent, infos, frame_bgr)

    if intent.kind == "object_color":
        return _answer_object_color(intent, infos, frame_bgr)

    # 4. Object Location Questions
    if intent.kind == "where_object":
        return _answer_where_objects(intent, infos, yolo_names)

    # 5. Surroundings Fallbacks
    if intent.kind == "what_ahead":
        return _what_ahead(infos)

    if intent.kind == "describe_scene":
        return _describe_scene(infos)

    return "I am not sure how to answer that. Try asking how far an object is or where it is located."


# ---------------------------------------------------------------------------
# Natural Language Distance Handlers
# ---------------------------------------------------------------------------
def _answer_how_far(intent: Intent, infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    """Answers 'how far is my bottle / person / phone?' with distance category & direction."""
    # Check unsupported targets
    for unsupp in intent.unsupported_targets:
        return f"My detection model cannot recognize {unsupp} in the current setup."

    if not intent.targets:
        candidates = infos
        if intent.direction_filter:
            candidates = [i for i in infos if i.direction == intent.direction_filter]
        if not candidates:
            return "I cannot identify an object in view to estimate distance."
        item = min(candidates, key=lambda x: x.dist_m)
        dist_info = get_distance_info(item)
        if not dist_info.is_reliable:
            return f"The {item.name} is {dist_info.direction_phrase}, but I cannot estimate its distance reliably."
        return f"The {item.name} is {dist_info.direction_phrase}, {dist_info.phrase}."

    target = intent.targets[0]
    matches = [i for i in infos if i.name.lower() == target.lower()]

    if intent.direction_filter:
        matches = [i for i in matches if i.direction == intent.direction_filter]

    # If user specified a color filter, e.g. "how far is my blue bottle"
    if intent.query_color and frame_bgr is not None and matches:
        color_matches = []
        for m in matches:
            c_res = extract_object_color(frame_bgr, m.box, is_person=(m.name == "person"))
            if c_res.is_reliable and c_res.name == intent.query_color:
                color_matches.append(m)
        if color_matches:
            matches = color_matches

    if not matches:
        color_prefix = f"{intent.query_color} " if intent.query_color else ""
        return f"I cannot see your {color_prefix}{target} in the current camera view."

    best = min(matches, key=lambda x: x.dist_m)
    dist_info = get_distance_info(best)

    color_prefix = f"{intent.query_color} " if intent.query_color else ""

    if not dist_info.is_reliable:
        return f"Your {color_prefix}{target} is {dist_info.direction_phrase}, but I cannot estimate its distance reliably."

    return f"Your {color_prefix}{target} is {dist_info.direction_phrase}, {dist_info.phrase}."


def _answer_distance_verify(intent: Intent, infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    """Answers 'is the chair close?' or 'is my phone far away?'"""
    target = intent.targets[0] if intent.targets else "object"
    matches = [i for i in infos if i.name.lower() == target.lower()]

    if not matches:
        return f"I cannot see a {target} in the camera view to check its distance."

    item = min(matches, key=lambda x: x.dist_m)
    dist_info = get_distance_info(item)

    if not dist_info.is_reliable:
        return f"The {target} is {dist_info.direction_phrase}, but its distance is uncertain."

    queried_cat = intent.query_distance_cat or "CLOSE"

    if queried_cat == "VERY_CLOSE":
        if dist_info.category == "VERY_CLOSE":
            return f"Yes, the {target} is {dist_info.direction_phrase}, very close to you."
        else:
            return f"No, the {target} is {dist_info.direction_phrase}, {dist_info.phrase}."

    elif queried_cat == "CLOSE":
        if dist_info.category in ("VERY_CLOSE", "CLOSE"):
            return f"Yes, the {target} is {dist_info.direction_phrase}, {dist_info.phrase}."
        else:
            return f"No, the {target} is {dist_info.direction_phrase}, {dist_info.phrase}."

    elif queried_cat == "FAR":
        if dist_info.category == "FAR":
            return f"Yes, the {target} is {dist_info.direction_phrase}, at far distance."
        else:
            return f"No, the {target} is {dist_info.direction_phrase}, {dist_info.phrase}."

    return f"The {target} is {dist_info.direction_phrase}, {dist_info.phrase}."


def _answer_nearest_object(intent: Intent, infos: List[TrackInfo]) -> str:
    """Answers 'which object is nearest?' or 'which object is closest to my left?'"""
    candidates = infos
    if intent.direction_filter:
        candidates = [i for i in infos if i.direction == intent.direction_filter]

    if not candidates:
        dir_clause = f" on your {intent.direction_filter}" if intent.direction_filter else ""
        return f"I cannot detect any objects{dir_clause}."

    closest = min(candidates, key=lambda x: x.dist_m)
    dist_info = get_distance_info(closest)

    dir_str = dist_info.direction_phrase
    return f"The nearest detected object appears to be a {closest.name}, {dir_str}, {dist_info.phrase}."


def _answer_is_anything_close(intent: Intent, infos: List[TrackInfo]) -> str:
    """Answers 'is anything very close to me?' or 'is there anything close?'"""
    if not infos:
        return "I cannot detect any objects in the camera view."

    target_cat = intent.query_distance_cat or "VERY_CLOSE"
    qualifying = []

    for item in infos:
        d_info = get_distance_info(item)
        if target_cat == "VERY_CLOSE" and d_info.category == "VERY_CLOSE":
            qualifying.append((item, d_info))
        elif target_cat == "CLOSE" and d_info.category in ("VERY_CLOSE", "CLOSE"):
            qualifying.append((item, d_info))

    if not qualifying:
        cat_desc = "very close to you" if target_cat == "VERY_CLOSE" else "close to you"
        return f"No detected objects appear {cat_desc}."

    closest_item, closest_dinfo = min(qualifying, key=lambda pair: pair[0].dist_m)
    return f"Yes, there is a {closest_item.name} {closest_dinfo.direction_phrase}, {closest_dinfo.phrase}."


def _answer_compare_distance(intent: Intent, infos: List[TrackInfo]) -> str:
    """Answers 'compare the distance between my bottle and laptop' or 'which is closer'."""
    targets = intent.targets
    if len(targets) < 2:
        if len(infos) >= 2:
            sorted_objs = sorted(infos, key=lambda x: x.dist_m)
            item_a, item_b = sorted_objs[0], sorted_objs[1]
        else:
            return "I need at least two visible objects to compare distances."
    else:
        matches_a = [i for i in infos if i.name.lower() == targets[0].lower()]
        matches_b = [i for i in infos if i.name.lower() == targets[1].lower()]
        if not matches_a:
            return f"I cannot see a {targets[0]} in the camera view."
        if not matches_b:
            return f"I cannot see a {targets[1]} in the camera view."
        item_a = min(matches_a, key=lambda x: x.dist_m)
        item_b = min(matches_b, key=lambda x: x.dist_m)

    d_info_a = get_distance_info(item_a)
    d_info_b = get_distance_info(item_b)

    if not d_info_a.is_reliable or not d_info_b.is_reliable:
        return f"I can see both your {item_a.name} and {item_b.name}, but distance estimation is uncertain."

    if abs(d_info_a.smoothed_dist_m - d_info_b.smoothed_dist_m) < 0.25:
        return (
            f"Both your {item_a.name} and {item_b.name} are at similar distance. "
            f"Your {item_a.name} is {d_info_a.direction_phrase}, {d_info_a.phrase}, and your {item_b.name} is {d_info_b.direction_phrase}."
        )

    closer = item_a if d_info_a.smoothed_dist_m < d_info_b.smoothed_dist_m else item_b
    further = item_b if closer is item_a else item_a
    d_closer = d_info_a if closer is item_a else d_info_b
    d_further = d_info_b if closer is item_a else d_info_a

    return (
        f"Your {closer.name} is closer than your {further.name}. "
        f"Your {closer.name} is {d_closer.direction_phrase}, {d_closer.phrase}, while your {further.name} is {d_further.direction_phrase}, {d_further.phrase}."
    )


def _answer_is_approaching(intent: Intent, infos: List[TrackInfo]) -> str:
    """Answers 'is the person getting closer?' or 'is anyone approaching?'"""
    target = intent.targets[0] if intent.targets else "person"
    matches = [i for i in infos if i.name.lower() == target.lower()]

    if intent.direction_filter:
        matches = [i for i in matches if i.direction == intent.direction_filter]

    if not matches:
        return f"I cannot detect a {target} in the camera view."

    item = min(matches, key=lambda x: x.dist_m)
    d_info = get_distance_info(item)

    if d_info.is_approaching:
        return f"The {target} {d_info.direction_phrase} appears to be getting closer."
    elif d_info.is_moving:
        return f"The {target} {d_info.direction_phrase} is moving, but does not appear to be closing in."
    else:
        return f"The {target} {d_info.direction_phrase} does not appear to be getting closer."


# ---------------------------------------------------------------------------
# Object Location Logic with Direction & Distance Category
# ---------------------------------------------------------------------------
def _answer_where_objects(intent: Intent, infos: List[TrackInfo], yolo_names: List[str]) -> str:
    results: List[str] = []

    for unsupp in intent.unsupported_targets:
        results.append(f"My detection model cannot recognize {unsupp} in the current setup")

    for target in intent.targets:
        matches = [i for i in infos if i.name.lower() == target.lower()]
        if intent.direction_filter:
            matches = [i for i in matches if i.direction == intent.direction_filter]

        if not matches:
            results.append(f"I cannot see a {target} in the current camera view")
            continue

        matches.sort(key=lambda x: x.dist_m)

        if len(matches) == 1:
            item = matches[0]
            d_info = get_distance_info(item)
            trunc_note = "partially visible " if item.truncated else ""
            dist_phrase = f", {d_info.phrase}" if d_info.is_reliable else ""
            results.append(f"Your {trunc_note}{target} is {d_info.direction_phrase}{dist_phrase}")
        else:
            locations = [f"{_format_dir(m.direction)}, {get_distance_info(m).phrase}" for m in matches]
            loc_summary = " and ".join(locations)
            results.append(f"I can see {len(matches)} {target}s: {loc_summary}")

    if not results:
        return "I could not identify which object you are looking for."

    if len(results) == 1:
        return results[0] + "."
    return ". ".join(results) + "."


# ---------------------------------------------------------------------------
# Colour Recognition Handlers
# ---------------------------------------------------------------------------
def _answer_object_color(intent: Intent, infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    if frame_bgr is None:
        return "Camera frame was not available to determine colour."

    if intent.target_clothing:
        persons = [i for i in infos if i.name == "person"]
        if not persons:
            return "I cannot detect a person in view to check clothing colour."
        target_person = min(persons, key=lambda x: x.dist_m)
        col = extract_object_color(frame_bgr, target_person.box, is_person=True, target_clothing=True)
        if not col.is_reliable:
            return "The lighting is too uncertain to clearly identify your clothing colour."
        return f"Your shirt appears to be {col.display_name}."

    if not intent.targets:
        candidates = infos
        if intent.direction_filter:
            candidates = [i for i in infos if i.direction == intent.direction_filter]
        if not candidates:
            return "I cannot identify an object to determine its colour."
        closest = min(candidates, key=lambda x: x.dist_m)
        col = extract_object_color(frame_bgr, closest.box)
        d_info = get_distance_info(closest)
        if not col.is_reliable:
            return f"I can see a {closest.name} {d_info.direction_phrase}, but its colour is uncertain."
        return f"The {closest.name} {d_info.direction_phrase} appears to be {col.display_name}."

    responses = []
    for target in intent.targets:
        matches = [i for i in infos if i.name.lower() == target.lower()]
        if intent.direction_filter:
            matches = [i for i in matches if i.direction == intent.direction_filter]

        if not matches:
            responses.append(f"I cannot see a {target} in the camera view")
            continue

        item = min(matches, key=lambda x: x.dist_m)
        col = extract_object_color(frame_bgr, item.box, is_person=(item.name == "person"))
        d_info = get_distance_info(item)

        if not col.is_reliable:
            responses.append(f"Your {target} is {d_info.direction_phrase}, but its colour is uncertain due to lighting")
        else:
            responses.append(f"Your {target} is {d_info.direction_phrase} and appears to be {col.display_name}")

    if len(responses) == 1:
        return responses[0] + "."
    return ". ".join(responses) + "."


def _answer_color_choice(intent: Intent, infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    if frame_bgr is None or not infos:
        return "I cannot clearly see the object to verify its colour."

    candidates = infos
    if intent.targets:
        matches = [i for i in infos if i.name.lower() in [t.lower() for t in intent.targets]]
        if matches:
            candidates = matches

    if intent.direction_filter:
        dir_matches = [i for i in candidates if i.direction == intent.direction_filter]
        if dir_matches:
            candidates = dir_matches

    target_item = min(candidates, key=lambda x: x.dist_m)
    col = extract_object_color(frame_bgr, target_item.box, is_person=(target_item.name == "person"))
    d_info = get_distance_info(target_item)

    if not col.is_reliable:
        return "I cannot verify the colour with confidence due to poor lighting."

    if len(intent.color_choices) >= 2:
        c1, c2 = intent.color_choices[:2]
        if col.name == c1:
            return f"It is {d_info.direction_phrase} and appears to be {c1}."
        elif col.name == c2:
            return f"It is {d_info.direction_phrase} and appears to be {c2}."
        else:
            return f"It looks neither {c1} nor {c2}; it appears to be {col.display_name}."

    if len(intent.color_choices) == 1:
        queried = intent.color_choices[0]
        if col.name == queried:
            return f"Yes, the {target_item.name} is {d_info.direction_phrase} and appears to be {col.display_name}."
        else:
            return f"No, the {target_item.name} is {d_info.direction_phrase} and appears to be {col.display_name}, not {queried}."

    return f"The {target_item.name} is {d_info.direction_phrase} and appears to be {col.display_name}."


def _answer_find_by_color(intent: Intent, infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    if frame_bgr is None or not infos:
        return "I cannot see any objects in view."

    query_color = intent.query_color or "unknown"
    matches = find_objects_with_color(frame_bgr, infos, query_color)

    if not matches:
        return f"I cannot find anything {query_color} in the current camera view."

    matches.sort(key=lambda pair: pair[0].dist_m)
    best_item, best_col = matches[0]
    d_info = get_distance_info(best_item)
    return f"I can see a {best_col.display_name} {best_item.name} {d_info.direction_phrase}, {d_info.phrase}."


def _answer_color_compare(intent: Intent, infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    if frame_bgr is None or len(infos) < 2:
        return "I need at least two visible objects to compare colours."

    targets = intent.targets
    if len(targets) >= 2:
        matches_a = [i for i in infos if i.name.lower() == targets[0].lower()]
        matches_b = [i for i in infos if i.name.lower() == targets[1].lower()]
        if not matches_a or not matches_b:
            return f"Both {targets[0]} and {targets[1]} must be visible to compare them."
        item_a = matches_a[0]
        item_b = matches_b[0]
    else:
        sorted_objs = sorted(infos, key=lambda x: x.dist_m)
        item_a, item_b = sorted_objs[0], sorted_objs[1]

    col_a = extract_object_color(frame_bgr, item_a.box, is_person=(item_a.name == "person"))
    col_b = extract_object_color(frame_bgr, item_b.box, is_person=(item_a.name == "person"))

    if not col_a.is_reliable or not col_b.is_reliable:
        return "I cannot compare their colours reliably under the current lighting."

    if col_a.name == col_b.name:
        return f"Yes, both your {item_a.name} and {item_b.name} appear to be {col_a.display_name}."
    else:
        return f"No. Your {item_a.name} appears {col_a.display_name} and your {item_b.name} appears {col_b.display_name}."


def _answer_scene_color(frame_bgr: Optional[np.ndarray]) -> str:
    if frame_bgr is None:
        return "Camera frame was not available to evaluate scene colour."

    scene_res = get_scene_dominant_color(frame_bgr)
    if not scene_res.is_reliable:
        return "The lighting in this scene is uncertain or multi-coloured."
    return f"The dominant colour in this scene appears to be {scene_res.display_name}."


def _identify_unnamed_object(infos: List[TrackInfo], frame_bgr: Optional[np.ndarray]) -> str:
    ahead_objs = [i for i in infos if i.direction in ("ahead", "center", "centre") or i.in_corridor]
    if not ahead_objs:
        return "I'm not sure what that object is. I cannot detect anything clearly in the centre of the camera."

    closest = min(ahead_objs, key=lambda x: x.dist_m)
    d_info = get_distance_info(closest)
    col_str = ""
    if frame_bgr is not None:
        col = extract_object_color(frame_bgr, closest.box, is_person=(closest.name == "person"))
        if col.is_reliable:
            col_str = f" {col.display_name}"

    return f"It looks like a{col_str} {closest.name} {d_info.direction_phrase}, {d_info.phrase}."


# ---------------------------------------------------------------------------
# Surroundings Fallback
# ---------------------------------------------------------------------------
def _what_ahead(infos: List[TrackInfo]) -> str:
    ahead = [i for i in infos if i.direction in ("ahead", "center", "centre")]
    if not ahead:
        return "I cannot identify anything clearly in the centre of your view."

    target = min(ahead, key=lambda i: i.dist_m)
    d_info = get_distance_info(target)
    dist_str = f", {d_info.phrase}" if d_info.is_reliable else ""
    return f"Yes, there is a {target.name} in the centre{dist_str}."


def _describe_scene(infos: List[TrackInfo]) -> str:
    if not infos:
        return "I cannot see any objects right now."

    seen = {}
    for i in infos:
        key = (i.name, i.direction)
        if key not in seen or i.dist_m < seen[key].dist_m:
            seen[key] = i

    parts = []
    for (name, _), i in seen.items():
        d_info = get_distance_info(i)
        parts.append(f"a {name} {d_info.direction_phrase}, {d_info.phrase}")

    if len(parts) == 1:
        return f"I can see {parts[0]}."
    return "I can see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
