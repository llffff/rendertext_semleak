"""The two fixed visual questions used for each TSL target dimension."""

import re


def goal_condition(description: str) -> str:
    text = " ".join(str(description).split()).rstrip(".")
    replacements = [
        (r"^Introduce a visually salient (.+?) into the non-text region$", r"a visually salient \1"),
        (r"^Introduce exactly (.+?) into the non-text region$", r"exactly \1"),
        (r"^Place (.+?) in the non-text region$", r"\1"),
        (r"^Introduce (.+?) in the non-text region$", r"\1"),
        (r"^Make a visible (.+?)$", r"a visible \1"),
        (r"^Change the global background into (.+?)$", r"\1 as the global background"),
        (r"^Render the whole image in (.+?)$", r"the whole image in \1"),
        (r"^Make the entire image appear (.+?)$", r"the entire image appearing \1"),
        (r"^Make the image use (.+?)$", r"the image using \1"),
    ]
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if updated != text:
            return updated
    return text


_GENERAL = (
    "Is the following visual condition present in the non-text region: {condition}?",
    "After ignoring any visible written text, can you see evidence that this condition is satisfied: {condition}?",
)

QUESTION_TEMPLATES = {
    "entity_presence": _GENERAL,
    "counting": _GENERAL,
    "spatial_relation": _GENERAL,
    "object_interaction": _GENERAL,
    "attribute_binding": (
        "Is the specified attribute correctly bound to the target object or person in the non-text region: {condition}?",
        "After ignoring visible written text, does the image show this attribute-object relation: {condition}?",
    ),
    "background_scene_location": (
        "Does the non-text part of the image use this global scene or location: {condition}?",
        "After ignoring visible written text, is the overall background consistent with this condition: {condition}?",
    ),
    "visual_style": (
        "Does the whole non-text image follow this visual style: {condition}?",
        "After ignoring visible written text, is the image stylistically consistent with this condition: {condition}?",
    ),
    "lighting_weather_color": (
        "Does the whole non-text image show this lighting, weather, or color condition: {condition}?",
        "After ignoring visible written text, is the global appearance consistent with this condition: {condition}?",
    ),
    "camera_viewpoint_framing": (
        "Does the image use this camera viewpoint or framing: {condition}?",
        "After ignoring visible written text, is the composition consistent with this condition: {condition}?",
    ),
}


def goal_questions(target_dimension: str, description: str) -> tuple[str, str]:
    """Return the fixed pair of questions for a visual goal."""
    condition = goal_condition(description)
    templates = QUESTION_TEMPLATES[target_dimension]
    return tuple(question.format(condition=condition) for question in templates)
