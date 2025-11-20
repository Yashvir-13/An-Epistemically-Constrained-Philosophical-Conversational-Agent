# Initial_priors.py
"""
Builds global categorical priors per axis from 'enriched_documents.json'.
We DO NOT flip signs for 'not_*' labels. We map concepts to (axis, category)
using Heirarchial_Splitting.map_concept_to_axis_category.
"""

import json
from collections import defaultdict

from Heirarchial_Splitting import AXES_SPEC, map_concept_to_axis_category

INPUT_FILE = "enriched_documents.json"

def build_global_priors(input_file: str = INPUT_FILE):
    """
    Returns:
      priors_by_axis = {
        axis_name: { category: prob, ... }
      }
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'.")
        return {}

    # Accumulate weighted evidence
    accum = {axis: defaultdict(float) for axis in AXES_SPEC.keys()}

    for d in docs:
        meta = d.get('metadata', {})
        belief_state = meta.get('belief_state', {})
        if isinstance(belief_state, str):
            try:
                belief_state = json.loads(belief_state)
            except:
                belief_state = {}

        for concept, val in belief_state.items():
            if not isinstance(val, (int, float)):
                continue
            axis, cat = map_concept_to_axis_category(concept)
            if axis and cat and cat in AXES_SPEC[axis]:
                # Convert val (-1..1) → weight (0..1) by affine map & ReLU
                weight = max(0.0, (val + 1.0) / 2.0)
                accum[axis][cat] += weight

    # Normalize to probability distributions
    priors_by_axis = {}
    for axis, cat_counts in accum.items():
        total = sum(cat_counts.values()) or 1.0
        priors_by_axis[axis] = {cat: (cat_counts.get(cat, 0.0) / total) for cat in AXES_SPEC[axis]}

        # If everything is zero (no evidence), set uniform
        if sum(priors_by_axis[axis].values()) == 0.0:
            uniform = 1.0 / len(AXES_SPEC[axis])
            priors_by_axis[axis] = {cat: uniform for cat in AXES_SPEC[axis]}

    return priors_by_axis
