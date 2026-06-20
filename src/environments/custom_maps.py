"""Custom grid-world environments for loop and dead-end analysis."""

from __future__ import annotations

CUSTOM_LOOP_MAP = [
    "SFFFFFFF",
    "FHFHFFHF",
    "FFFHFFHF",
    "FHHFFFHF",
    "FFFFHFFF",
    "HFHFFFHF",
    "FFHFFHFF",
    "FFFHFFFG",
]

CUSTOM_LOOP_DESC = (
    "Custom 8x8 maze with loops, dead ends, and multiple alternative paths."
)
