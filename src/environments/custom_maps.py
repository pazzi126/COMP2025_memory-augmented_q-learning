"""Custom grid-world environments for loop and dead-end analysis."""

from __future__ import annotations

CUSTOM_LOOP_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

CUSTOM_LOOP_DESC = (
    "Custom 4x4 map with loops and a hole (H) to encourage repeated-state behaviour."
)
