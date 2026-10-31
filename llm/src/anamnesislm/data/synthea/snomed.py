"""SNOMED CT name helpers.

SNOMED fully specified names end with a semantic tag naming the concept's hierarchy,
e.g. "Diabetes mellitus type 2 (disorder)" or "Eggs (edible) (substance)". Not every
trailing parenthesis is a tag: Synthea's dental conditions use names such as
"Fracture of tooth (traumatic)", so only known tags are split off.
"""

import re

SEMANTIC_TAGS = frozenset(
    {
        "assessment scale",
        "attribute",
        "body structure",
        "cell",
        "cell structure",
        "clinical drug",
        "disorder",
        "environment",
        "ethnic group",
        "event",
        "finding",
        "life style",
        "medicinal product",
        "morphologic abnormality",
        "observable entity",
        "occupation",
        "organism",
        "person",
        "physical force",
        "physical object",
        "procedure",
        "product",
        "qualifier value",
        "record artifact",
        "regime/therapy",
        "religion/philosophy",
        "situation",
        "social concept",
        "specimen",
        "substance",
    }
)

_TRAILING_PARENTHESIS = re.compile(r"^(?P<name>.*\S)\s*\((?P<tag>[^()]+)\)$")


def split_semantic_tag(label: str) -> tuple[str, str | None]:
    """Split "Name (tag)" into ("Name", "tag"); return (label, None) if there is no known tag."""
    match = _TRAILING_PARENTHESIS.match(label.strip())
    if match is None or match["tag"] not in SEMANTIC_TAGS:
        return label.strip(), None
    return match["name"], match["tag"]
