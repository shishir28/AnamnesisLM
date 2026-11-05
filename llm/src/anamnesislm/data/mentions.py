"""Made-up and shortened names, so the model learns to copy names instead of memorising them.

Synthea gives only ~280 distinct names, always spelled the same way, so a model can recall
them instead of copying from the note. Swapping in names it cannot have seen forces copying.

Two disjoint syllable sets: TRAIN_SYLLABLES builds names for training data, UNSEEN_SYLLABLES
builds names only for the test_unseen_names split, so those names never occur in training.
"""

import random
import re
from typing import Literal

NameSet = Literal["train", "unseen"]

TRAIN_SYLLABLES = [
    "ka", "lo", "mi", "ra", "ven", "tor", "sil", "pa", "zen", "dro",
    "mel", "fi", "ga", "nu", "bel", "cor", "tis", "var", "do", "le",
]  # fmt: skip
UNSEEN_SYLLABLES = [
    "qui", "xo", "jum", "wex", "hob", "uz", "plo", "kry", "fen", "yab",
    "gri", "sov", "thu", "mox", "bry", "zul", "eck", "vom", "hix", "op",
]  # fmt: skip

DRUG_SUFFIXES = ["pril", "statin", "olol", "mab", "azole", "cillin", "ine", "afil"]
DOSES_MG = [5, 10, 20, 25, 50, 100, 250, 500]
CONDITION_PATTERNS = ["{} syndrome", "{} disease", "{}itis", "Chronic {}osis", "{} disorder"]
ALLERGEN_PATTERNS = ["{}", "{} extract", "{} pollen"]


def _stem(rng: random.Random, name_set: NameSet) -> str:
    syllables = TRAIN_SYLLABLES if name_set == "train" else UNSEEN_SYLLABLES
    return "".join(rng.choice(syllables) for _ in range(rng.randint(2, 3)))


def made_up_name(kind: str, rng: random.Random, name_set: NameSet) -> str:
    """A pronounceable name shaped like the kind, e.g. "Venloazole 20 MG Oral Tablet"."""
    stem = _stem(rng, name_set)
    if kind == "medication":
        drug = (stem + rng.choice(DRUG_SUFFIXES)).capitalize()
        return f"{drug} {rng.choice(DOSES_MG)} MG Oral Tablet" if rng.random() < 0.5 else drug
    patterns = CONDITION_PATTERNS if kind == "condition" else ALLERGEN_PATTERNS
    name = rng.choice(patterns).format(stem)
    return name[0].upper() + name[1:]


def short_drug_name(drug: str) -> str:
    """Drop the strength and form: "Metformin hydrochloride 500 MG Oral Tablet" ->
    "Metformin hydrochloride". Returns drug unchanged when there is no strength to drop."""
    match = re.match(r"^(\D+?)\s+\d", drug)
    return match.group(1) if match else drug
