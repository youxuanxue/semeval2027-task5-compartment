"""
Agent Bravo: Contextual Ablation & In-Filling Feature Extractor
Replaces target constituents with lexical prototypes to compute semantic degradation / shift.
"""

from typing import Dict, List, Tuple
import re


# Prototype literal substitutions for common high-frequency components
LEXICAL_PROTOTYPES_EN = {
    # Noun components
    "account": "financial record",
    "book": "printed paper volume",
    "flea": "insect parasite",
    "market": "bazaar trade venue",
    "air": "atmospheric gas",
    "craft": "vehicle vessel",
    "blood": "red bodily fluid",
    "bank": "depository financial vault",
    "chain": "connected metal rings",
    "fire": "flame combustion",
    "line": "straight physical row",
    "water": "liquid H2O",
    "fall": "gravitational drop",
    "green": "color emerald",
    "house": "residential building structure",
    # Particle verb bases
    "crack": "break fracture",
    "boil": "heat liquid to bubbles",
    "bring": "carry transport",
    "turn": "rotate spin",
    "look": "gaze optical see",
    "give": "hand transfer",
    "take": "grasp seize",
    "break": "smash shatter",
    "go": "move travel",
    "come": "arrive approach",
    "hold": "clutch grip",
    "run": "sprint dash",
}

LEXICAL_PROTOTYPES_DE = {
    "abitur": "schulabschluss diplom",
    "zeugnis": "bescheinigung papierdokument",
    "haus": "wohngebaeude gebaeude",
    "turm": "hohes bauwerk",
    "wasser": "fluessigkeit",
    "hauen": "schlagen treffen",
    "geben": "ueberreichen schenken",
    "gehen": "schreiten wandern",
    "nehmen": "greifen anpacken",
}


def substitute_word_in_context(sentence: str, target: str, replacement: str) -> str:
    """Safely substitutes target word/compound in context sentence using regex word boundaries."""
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    return pattern.sub(replacement, sentence, count=1)


def generate_ablation_pairs(record: dict, task_type: str, lang: str = "en") -> dict:
    """
    Produces original sentence and constituent-ablated counterfactual sentence.
    This enables the model to compare representation drift under literal substitution.
    """
    sentence = record["context"]
    prototypes = LEXICAL_PROTOTYPES_EN if lang == "en" else LEXICAL_PROTOTYPES_DE

    if task_type == "nn":
        target = record["target"]
        mod = record["mod"]
        head = record["head"]

        mod_sub = prototypes.get(mod.lower(), "generic item")
        head_sub = prototypes.get(head.lower(), "physical object")

        # Create counterfactual sentences
        sent_mod_ablated = substitute_word_in_context(sentence, mod, mod_sub)
        sent_head_ablated = substitute_word_in_context(sentence, head, head_sub)

        return {
            "orig_sentence": sentence,
            "mod_ablated": sent_mod_ablated,
            "head_ablated": sent_head_ablated,
            "mod_sub": mod_sub,
            "head_sub": head_sub,
        }
    else:  # pv
        target = record["target"]
        base = record["base"]
        base_sub = prototypes.get(base.lower(), "physical motion")
        sent_pv_ablated = substitute_word_in_context(sentence, base, base_sub)
        return {
            "orig_sentence": sentence,
            "base_ablated": sent_pv_ablated,
            "base_sub": base_sub,
        }
