"""Verified editorial lessons learned from prior narrative failures."""

DEFAULT_EDITORIAL_LESSONS: tuple[str, ...] = (
    "Character names are immutable facts. Never infer a renamed character from surrounding translated words.",
    "Before calling a detail invented, compare it against the source checklist; source-supported details are not inventions.",
    "Never mark a required event missing when the script contains evidence for that event.",
    "A repair must make a concrete change for every failed fact ID; do not repeat an unchanged script.",
    "When a targeted repair repeats the same script, switch strategy and rebuild from the verified fact checklist.",
    "Do not trust self-review alone for cross-language fact preservation; require independent semantic evidence.",
)
