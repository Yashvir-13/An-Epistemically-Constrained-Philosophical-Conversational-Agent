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
    Constructs Bayesian Priors from the ingested corpus (`enriched_documents.json`).
    
    It scans the entire database of enriched documents to see how frequently 
    concepts appear. This establishes the 'Base Rate' for each philosophical category.
    
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

    # Normalize to probability distributions with DAMPENING (Smoothing)
    priors_by_axis = {}
    DAMPING_FACTOR = 0.5 # Mix 50% uniform, 50% data derived
    
    for axis, cat_counts in accum.items():
        total = sum(cat_counts.values()) or 1.0
        data_dist = {cat: (cat_counts.get(cat, 0.0) / total) for cat in AXES_SPEC[axis]}
        
        # Mix with uniform
        uniform_prob = 1.0 / len(AXES_SPEC[axis])
        final_dist = {}
        for cat, p in data_dist.items():
            final_dist[cat] = (p * (1 - DAMPING_FACTOR)) + (uniform_prob * DAMPING_FACTOR)
            
        priors_by_axis[axis] = final_dist

    return priors_by_axis
