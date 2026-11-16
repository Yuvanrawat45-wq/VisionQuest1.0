"""Natural language intent parser for VisionQuest voice interactions.

Understands:
1. Distance & Proximity questions:
   - "How far is my bottle / person in front of me / phone?"
   - "Is the chair close?", "Is my phone far away?", "Is anything very close to me?"
   - "Which object is nearest?", "Which object is closest to my left?"
   - "Compare the distance between my bottle and laptop."
   - "Is the person getting closer?"
2. Colour recognition questions:
   - "What colour is my bottle / shirt / car?"
   - "Is this object red or blue?"
   - "Find something green."
   - "What colour is the bag on my left?"
   - "Are my bottle and bag the same colour?"
   - "What is the dominant colour in this scene?"
3. Object location questions (synonyms, single & multiple targets)
4. Unsupported object detection (keys, wallet, shoes, doors, etc.)
5. Commands: repeat, silence, resume, help, unknown object identification
6. Surroundings, proximity, people, and scene intents
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Supported Base Color Vocabulary
# ---------------------------------------------------------------------------
COLOR_VOCAB = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "black", "white", "grey", "gray"
}

CLOTHING_WORDS = {
    "shirt", "tshirt", "t-shirt", "top", "clothes", "clothing",
    "jacket", "coat", "hoodie", "sweater", "dress"
}

# ---------------------------------------------------------------------------
# Intent Data Model
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    kind: str                                  # 'where_object', 'how_far', 'object_color', etc.
    targets: List[str] = field(default_factory=list)  # Normalised YOLO class names
    unsupported_targets: List[str] = field(default_factory=list)  # Valid concepts not in YOLO
    raw_query: str = ""                        # Original user transcript
    direction_filter: Optional[str] = None     # 'left', 'right', 'ahead' (centre)
    query_color: Optional[str] = None          # e.g., 'blue' for 'how far is my blue bottle'
    color_choices: List[str] = field(default_factory=list)  # e.g., ['red', 'blue']
    target_clothing: bool = False              # True if asking about shirt / clothing
    is_comparison: bool = False                # True if comparing distance or color
    query_distance_cat: Optional[str] = None   # 'VERY_CLOSE', 'CLOSE', 'MEDIUM', 'FAR'


# ---------------------------------------------------------------------------
# Synonyms & Concept Mapping
# ---------------------------------------------------------------------------
SYNONYM_MAP = {
    "phone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "iphone": "cell phone",
    "cellphone": "cell phone",
    "water bottle": "bottle",
    "waterbottle": "bottle",
    "flask": "bottle",
    "tumbler": "bottle",
    "mug": "cup",
    "coffee cup": "cup",
    "teacup": "cup",
    "bag": "backpack",
    "schoolbag": "backpack",
    "handbag": "handbag",
    "purse": "handbag",
    "briefcase": "suitcase",
    "luggage": "suitcase",
    "sofa": "couch",
    "armchair": "chair",
    "seat": "chair",
    "stool": "chair",
    "tv": "tv",
    "television": "tv",
    "monitor": "tv",
    "computer screen": "tv",
    "desktop": "laptop",
    "notebook": "laptop",
    "computer": "laptop",
    "pc": "laptop",
    "spectacles": "glasses",
    "puppy": "dog",
    "kitten": "cat",
    "automobile": "car",
    "vehicle": "car",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
    "desk": "dining table",
    "table": "dining table",
}

UNSUPPORTED_OBJECTS = {
    "key", "keys", "door", "doors", "shoe", "shoes", "wallet", "wallets",
    "pen", "pens", "pencil", "pencils", "glasses", "sunglasses", "stair",
    "stairs", "staircase", "watch", "wristwatch", "charger", "cable",
    "earphones", "headphones", "slipper", "slippers", "window", "fan"
}

# ---------------------------------------------------------------------------
# Keyword Clusters for Parsing
# ---------------------------------------------------------------------------
_REPEAT_TRIGGERS = [
    "repeat", "repeat that", "say again", "say that again",
    "what did you say", "pardon", "could you repeat"
]

_SILENCE_TRIGGERS = [
    "be quiet", "shut up", "stop talking", "mute", "silence",
    "quiet please", "pause audio", "be silent"
]

_RESUME_TRIGGERS = [
    "start listening again", "start listening", "unmute", "resume",
    "continue", "you can speak", "listen again"
]

_HELP_TRIGGERS = [
    "what can you do", "help", "help me", "what commands",
    "how do i use this", "what can i ask"
]

_LOCATION_PHRASES = [
    "where is", "where are", "where did i leave", "where did i put",
    "can you find", "find my", "find the", "locate", "can you locate",
    "can you see", "is my", "is there a", "is there an", "are there any",
    "is the", "look for", "search for", "visible", "any sign of"
]


def _clean_text(text: str) -> str:
    """Normalize text: lower-case, remove punctuation except spaces."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _extract_colors_from_text(cleaned: str) -> List[str]:
    """Extract mentioned color names from query."""
    found = []
    for word in cleaned.split():
        if word in COLOR_VOCAB:
            c = "grey" if word == "gray" else word
            if c not in found:
                found.append(c)
    return found


def _extract_direction_filter(cleaned: str) -> Optional[str]:
    """Detect if query specifically targets 'left', 'right', or 'ahead/front/centre'."""
    if "left" in cleaned:
        return "left"
    elif "right" in cleaned:
        return "right"
    elif "front" in cleaned or "ahead" in cleaned or "centre" in cleaned or "center" in cleaned:
        return "ahead"
    return None


def _extract_objects_from_text(cleaned: str, yolo_names: List[str]):
    """Extract recognized YOLO classes and unsupported classes mentioned in text."""
    found_yolo: Set[str] = set()
    found_unsupported: Set[str] = set()

    for phrase, target in sorted(SYNONYM_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, cleaned):
            if target in yolo_names:
                found_yolo.add(target)
            elif target in UNSUPPORTED_OBJECTS:
                found_unsupported.add(phrase)

    for name in sorted(yolo_names, key=len, reverse=True):
        pattern = r"\b" + re.escape(name.lower()) + r"(?:s|es)?\b"
        if re.search(pattern, cleaned):
            found_yolo.add(name)

    for item in sorted(UNSUPPORTED_OBJECTS, key=len, reverse=True):
        pattern = r"\b" + re.escape(item) + r"\b"
        if re.search(pattern, cleaned):
            if not any(y in item for y in found_yolo):
                found_unsupported.add(item)

    return list(found_yolo), list(found_unsupported)


# ---------------------------------------------------------------------------
# Main Parse Function
# ---------------------------------------------------------------------------
def parse(transcript: str, yolo_names: List[str]) -> Intent:
    """Parse a speech transcript into a structured Intent object."""
    if not transcript or not transcript.strip():
        return Intent(kind="unknown", raw_query=transcript or "")

    cleaned = _clean_text(transcript)

    # 1. System Control Commands
    if any(cleaned == trig or cleaned.startswith(trig) for trig in _REPEAT_TRIGGERS):
        return Intent(kind="repeat", raw_query=transcript)

    if any(cleaned == trig or trig in cleaned for trig in _SILENCE_TRIGGERS):
        return Intent(kind="silence", raw_query=transcript)

    if any(cleaned == trig or trig in cleaned for trig in _RESUME_TRIGGERS):
        return Intent(kind="resume", raw_query=transcript)

    if any(cleaned == trig or trig in cleaned for trig in _HELP_TRIGGERS):
        return Intent(kind="help", raw_query=transcript)

    # 2. Unknown object identification
    if ("what is that" in cleaned or "what is this" in cleaned or
            "what is in my hand" in cleaned or "identify that" in cleaned):
        return Intent(kind="identify_object", raw_query=transcript)

    # Extract common elements
    colors_found = _extract_colors_from_text(cleaned)
    direction_filter = _extract_direction_filter(cleaned)
    is_clothing = any(cw in cleaned for cw in CLOTHING_WORDS)
    yolo_matches, unsupported_matches = _extract_objects_from_text(cleaned, yolo_names)

    # -----------------------------------------------------------------------
    # 3. DISTANCE & PROXIMITY INTENTS
    # -----------------------------------------------------------------------

    # A. Approaching / Motion check: "is the person getting closer?", "is anyone approaching"
    if any(phrase in cleaned for phrase in ["getting closer", "approaching", "moving toward", "moving towards", "coming closer"]):
        target = yolo_matches if yolo_matches else ["person"]
        return Intent(
            kind="is_approaching",
            targets=target,
            direction_filter=direction_filter,
            raw_query=transcript
        )

    # B. Compare distance: "compare the distance between my bottle and laptop", "which is closer"
    if ("compare" in cleaned and "distance" in cleaned) or "which is closer" in cleaned or "closer than" in cleaned:
        return Intent(
            kind="compare_distance",
            targets=yolo_matches,
            is_comparison=True,
            raw_query=transcript
        )

    # C. Nearest object: "which object is nearest?", "which object is closest to my left?", "what is closest"
    if any(w in cleaned for w in ["nearest", "closest"]):
        return Intent(
            kind="nearest_object",
            targets=yolo_matches,
            direction_filter=direction_filter,
            raw_query=transcript
        )

    # D. How far: "how far is my bottle?", "how far is the person in front of me?", "how far is my blue bottle"
    if "how far" in cleaned or "what distance" in cleaned or "how close" in cleaned:
        return Intent(
            kind="how_far",
            targets=yolo_matches,
            unsupported_targets=unsupported_matches,
            query_color=colors_found[0] if colors_found else None,
            direction_filter=direction_filter,
            raw_query=transcript
        )

    # E. Distance verification: "is the chair close?", "is anything very close to me?", "is my phone far away"
    if any(phrase in cleaned for phrase in ["very close", "far away", "far", "close", "near me"]):
        cat = "CLOSE"
        if "very close" in cleaned:
            cat = "VERY_CLOSE"
        elif "far" in cleaned or "far away" in cleaned:
            cat = "FAR"

        if "anything" in cleaned or "something" in cleaned or not yolo_matches:
            return Intent(
                kind="is_anything_close",
                query_distance_cat=cat,
                direction_filter=direction_filter,
                raw_query=transcript
            )

        return Intent(
            kind="is_distance_verify",
            targets=yolo_matches,
            unsupported_targets=unsupported_matches,
            query_distance_cat=cat,
            direction_filter=direction_filter,
            raw_query=transcript
        )

    # -----------------------------------------------------------------------
    # 4. COLOUR RECOGNITION INTENTS
    # -----------------------------------------------------------------------
    is_color_q = "color" in cleaned or "colour" in cleaned

    if is_color_q and any(w in cleaned for w in ["scene", "room", "dominant", "main color", "main colour", "environment"]):
        return Intent(kind="scene_color", raw_query=transcript)

    if (("find" in cleaned or "see" in cleaned or "show" in cleaned or "is there" in cleaned)
            and ("something" in cleaned or "anything" in cleaned or "object" in cleaned)
            and colors_found):
        return Intent(
            kind="find_by_color",
            query_color=colors_found[0],
            raw_query=transcript
        )

    if is_color_q and ("same" in cleaned or "compare" in cleaned or "different" in cleaned):
        return Intent(
            kind="color_compare",
            targets=yolo_matches,
            is_comparison=True,
            raw_query=transcript
        )

    if len(colors_found) >= 2 or (" or " in cleaned and len(colors_found) >= 1) or (cleaned.startswith("is ") and colors_found):
        target = yolo_matches if yolo_matches else (["person"] if is_clothing else [])
        return Intent(
            kind="color_choice",
            targets=target,
            color_choices=colors_found,
            target_clothing=is_clothing,
            direction_filter=direction_filter,
            raw_query=transcript
        )

    if is_color_q:
        if is_clothing and "person" in yolo_names:
            yolo_matches = ["person"]

        return Intent(
            kind="object_color",
            targets=yolo_matches,
            unsupported_targets=unsupported_matches,
            target_clothing=is_clothing,
            direction_filter=direction_filter,
            raw_query=transcript
        )

    # -----------------------------------------------------------------------
    # 5. OBJECT LOCATION INTENTS
    # -----------------------------------------------------------------------
    is_location_q = any(phrase in cleaned for phrase in _LOCATION_PHRASES)

    if is_location_q or yolo_matches or unsupported_matches:
        if yolo_matches or unsupported_matches:
            return Intent(
                kind="where_object",
                targets=yolo_matches,
                unsupported_targets=unsupported_matches,
                direction_filter=direction_filter,
                raw_query=transcript
            )

    # 6. Fallback for surroundings / ahead
    if any(w in cleaned for w in ["around me", "surroundings", "what can you see", "describe"]):
        return Intent(kind="describe_scene", raw_query=transcript)

    if any(w in cleaned for w in ["front of me", "ahead", "in front", "in my way"]):
        return Intent(kind="what_ahead", raw_query=transcript)

    return Intent(kind="unknown", raw_query=transcript)
