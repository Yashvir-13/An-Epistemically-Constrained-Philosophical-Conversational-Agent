from typing import List, Dict, Tuple, Literal

def belief_strength(p: float) -> Literal["strong", "moderate", "weak"]:
    """
    Classifies belief strength based on posterior probability.
    """
    if p >= 0.90:
        return "strong"
    elif p >= 0.70:
        return "moderate"
    else:
        return "weak"

def build_belief_constraints(graph: List[Dict], posteriors: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Dynamically generates 'Guardrails' based on the agent's current beliefs (Posteriors).
    
    Logic:
    1. **Strong Beliefs (>0.9)** -> Non-Negotiable Constraints ("You MUST NOT contradict...").
    2. **Onotological Rules**: If a specific category is believed (e.g. "Will-Lessness"), 
       it automatically triggers semantic polarity rules (e.g. "Forbid active verbs like 'causes'").
       
    Args:
        graph (list): Knowledge graph nodes.
        posteriors (dict): Current probabilities.
        
    Returns:
        tuple: (non_negotiable, negotiable, tensions, polarity_rules)
    """
    non_negotiable = []
    negotiable = []
    tensions = []
    polarity_rules = []

    for node in graph:
        nid = node["id"]
        axis = node.get("axis", "Unknown Axis")
        
        dist = posteriors.get(nid, {})
        if not dist:
            continue
            
        # Get top belief
        top_cat, prob = max(dist.items(), key=lambda x: x[1])
        strength = belief_strength(prob)
        
        belief_stmt = f"The {axis.replace('_', ' ')} is '{top_cat}'"
        
        # --- 1. PROPOSITIONAL CONSTRAINTS ---
        if strength == "strong":
            constraint = f"{belief_stmt} (Certainty: {prob:.2f}). Do not contradict this."
            non_negotiable.append(constraint)
        elif strength == "moderate":
            commitment = f"{belief_stmt} (Certainty: {prob:.2f}). Discuss with nuance."
            negotiable.append(commitment)
        
        # --- 2. SEMANTIC POLARITY GUARDS (LINGUISTIC DISCIPLINE) ---
        # Derive forbidden phrases from the ONTOLOGY of the strong belief.
        # This is strictly logical derivation, not Doctrine hardcoding.
        
        if strength == "strong":
            # If something is "will_less" or "denies_will", it cannot "affirm" or "cause".
            if top_cat in ["will_less", "denies_will", "will_less_contemplation", "suspension"]:
                 polarity_rules.append(
                     f"Because {axis} is '{top_cat}': FORBID words implying active will ('affirms', 'manifests', 'acts on', 'direct effect'). Use PASSIVE/NEGATIVE frames ('suspends', 'silences', 'free from')."
                 )
            
            # If something is "futile" or "suffering", it cannot be "good" or "redemptive" (unless specifically ascetic).
            if top_cat in ["futile", "suffering", "deluded"]:
                 polarity_rules.append(
                     f"Because {axis} is '{top_cat}': FORBID positive/redemptive verbs ('heals', 'solves', 'redeems'). Use PESSIMISTIC frames ('escapes momentarily', 'fails to satisfy')."
                 )

            # If something IS "affirms_will", it cannot be "liberating".
            if top_cat == "affirms_will":
                 polarity_rules.append(
                     f"Because {axis} is '{top_cat}': FORBID liberation language ('frees', 'saves', 'transcends'). Use ENTRAPMENT frames ('binds', 'entangles', 'feeds')."
                 )

    return non_negotiable, negotiable, tensions, polarity_rules
