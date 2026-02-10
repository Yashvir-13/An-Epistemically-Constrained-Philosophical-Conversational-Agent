# probabilistic_reasoner.py — Philosopher AI v2.4
# Refactored: Dynamic Soft Logic Network (Data-Driven)

import math
import json
import os
from typing import Dict, List

# Fallback / Default Matrix if file not found
DEFAULT_CORRELATIONS = {
    "metaphysical_status:affirms_will": {
        "liberation_status:futile": 1.5,
        "ethical_status:outside_morality": 1.5,
        "ethical_status:morally_wrong": -1.0, 
    },
    "metaphysical_status:denies_will": {
        "ethical_status:ascetic_good": 1.5,
        "liberation_status:liberating": 1.2,
    }
}

# Axis Coupling Strength (Scaling factors)
AXIS_COUPLING = {
    "metaphysical_status": 1.0,
    "ethical_status": 0.9,
    "psychological_cause": 0.6,
    "liberation_status": 0.9,
    "aesthetic_effect": 0.85, # RESTORED: Aesthetics is objective truth (Ideas), not just weak feeling
    "religious_alignment": 0.5
}

def _logit(p: float, epsilon: float = 1e-9) -> float:
    p = max(epsilon, min(1.0 - epsilon, p))
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a dictionary of probabilities to sum to 1.0"""
    if not dist:
        return {}
    s = sum(dist.values())
    if s == 0:
        return {k: 1.0/len(dist) for k in dist}
    return {k: v/s for k, v in dist.items()}

def calculate_entropy(dist: Dict[str, float]) -> float:
    """Calculates Shannon entropy of the distribution."""
    if not dist:
        return 0.0
    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

class EpistemicProfile:
    """
    Represents the agent's 'self-knowledge' about a specific belief node.
    
    Tracks:
    - **Confidence**: How strong is the top belief?
    - **Entropy**: How confused is the distribution? (0.0 = Certainty, High = Confusion)
    - **Tier**: STRONG (>0.9), MODERATE, or WEAK.
    - **Permissions**: What the agent is allowed to say (e.g., can it assert truth or only describe?).
    """
    def __init__(self, node_id, axis, dist):
        self.node_id = node_id
        self.axis = axis
        self.dist = dist
        self.entropy = calculate_entropy(dist)
        
        # Determine Top Belief & Confidence
        if dist:
            self.top_cat, self.confidence = max(dist.items(), key=lambda x: x[1])
        else:
            self.top_cat, self.confidence = "Unknown", 0.0
            
        # Determine Tier
        if self.confidence >= 0.90:
            self.tier = "STRONG"
        elif self.confidence >= 0.70:
            self.tier = "MODERATE"
        else:
            self.tier = "WEAK"
            
        # Determine Permissions
        self.permissions = {
            "assert": self.tier == "STRONG",
            "explain": True, # Always allowed to explain the concept
            "generalize": self.tier in ["STRONG", "MODERATE"],
            "metaphysical_claims": self.tier == "STRONG", # Only strong beliefs allow ontological assertions
            "causal_language": self.tier == "STRONG" and self.entropy < 0.6 # strict check
        }
        
    def to_dict(self):
        return {
            "top_cat": self.top_cat,
            "confidence": self.confidence,
            "tier": self.tier,
            "entropy": self.entropy,
            "permissions": self.permissions
        }

class SoftLogicNetwork:
    """
    A Probabilistic Soft Logic (PSL) Reasoner.
    
    Unlike standard Bayesian networks, this uses a **Logit-based fusion** approach:
    1. Converts Probabilities to Logits (Log-odds).
    2. Adds evidences and priors linearly (Weighted sum).
    3. Applies **Cross-Axis Correlations** as additive 'nudges' to the logits.
       (e.g., if Metaphysics=Affirmation, nudge Ethics towards Wrong).
    4. Converts back to Probability via Sigmoid.
    
    This allows for stable, differentiable-like updates without the explosion of full Bayesian inference.
    """

    def __init__(self, graph: List[Dict], priors_by_axis: Dict[str, Dict[str, float]],
                 axis_relevance_map: Dict[str, float] = None,
                 prior_weight: float = 1.0,
                 evidence_weight: float = 0.8,
                 correlation_strength: float = 20.0,
                 correlation_file: str = "learned_correlations.json"):
        
        self.graph = graph
        self.priors_by_axis = priors_by_axis or {}
        self.axis_relevance_map = axis_relevance_map or {}
        
        # New Weights
        self.prior_weight = float(prior_weight)
        self.evidence_weight = float(evidence_weight)
        self.correlation_strength = float(correlation_strength)
        
        # Debug Trace
        self.trace = {
            "applied_nudges": [],
            "logit_updates": []
        }

        # Load learned correlations
        if os.path.exists(correlation_file):
            try:
                with open(correlation_file, 'r') as f:
                    self.correlations = json.load(f)
            except Exception as e:
                print(f"   [SoftLogic] Error loading correlations: {e}. Using defaults.")
                self.correlations = DEFAULT_CORRELATIONS
        else:
            self.correlations = DEFAULT_CORRELATIONS

        # Index nodes by axis
        self.nodes_by_axis = {}
        for n in graph:
            ax = n.get("axis", "")
            if ax:
                self.nodes_by_axis.setdefault(ax, []).append(n["id"])

    def _node_prior(self, node: Dict) -> Dict[str, float]:
        axis = node.get("axis", "")
        cats = node.get("categories", [])
        px = self.priors_by_axis.get(axis, {})
        if not cats:
            return {}
        subset = {c: px.get(c, 0.0) for c in cats}
        return _normalize(subset)

    def _combine_prior_evidence(self, prior: Dict[str, float], evidence: Dict[str, float]) -> Dict[str, float]:
        posterior = {}
        CLIP = 12.0
        
        for c in prior.keys():
            p = prior.get(c, 0.5)
            e = evidence.get(c, 0.5) # 0.5 is neutral in logit space (0)
            
            # Convert to logit
            l_p = _logit(p)
            l_e = _logit(e)
            
            # Weighted Fusion
            l_final = (l_p * self.prior_weight) + (l_e * self.evidence_weight)
            
            # Clip
            l_final = max(-CLIP, min(CLIP, l_final))
            
            posterior[c] = _sigmoid(l_final)
        
        return _normalize(posterior)

    # --------- Cross-axis correlations (Logit Logic) ---------

    def _apply_correlations(self, posteriors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Applies additive logit nudges.
        """
        # We work with logits for updates, then convert back
        # Step 1: Convert current posteriors to logits
        logits = {}
        for nid, dist in posteriors.items():
            logits[nid] = {k: _logit(v) for k, v in dist.items()}

        # Step 2: Apply correlations as logit deltas
        # Identify source nodes
        for nid, dist in posteriors.items():
            if not dist: continue
            best_cat = max(dist.items(), key=lambda x: x[1])[0]
            
            node_ax = next((n["axis"] for n in self.graph if n["id"] == nid), None)
            if not node_ax: continue

            # CHECK RELEVANCE
            # Ensure correlations are applied only from axes whose axis_relevance score >= 0.15 
            # OR from the first/primary axis.
            # (We assume the first axis in ranking has highest score, check map)
            
            relevance = self.axis_relevance_map.get(node_ax, 0.0)
            # We also need to know if it's the "primary" axis. 
            # We can infer primary if relevance is max in map, or check strict value.
            # Simpler: if relevance < 0.15, we check if it's the max relevance in the map.
            
            is_primary = False
            if self.axis_relevance_map:
                max_rel = max(self.axis_relevance_map.values())
                if relevance == max_rel:
                    is_primary = True
            
            if relevance < 0.15 and not is_primary:
                # print(f"Skipping correlations from {node_ax} (relevance {relevance:.2f} < 0.15)")
                continue

            # Key for correlation rules
            key = f"{node_ax}:{best_cat}"
            node_correlations = self.correlations.get(key, {})

            # Apply Logic
            for target_key, raw_nudge in node_correlations.items():
                if ":" not in target_key: continue
                target_axis, target_cat = target_key.split(":")
                
                # Get coupling factor
                coupling = AXIS_COUPLING.get(target_axis, 1.0)
                
                # Calculate Logit Delta
                # logit_delta = strength * coupling * nudge
                logit_delta = self.correlation_strength * coupling * raw_nudge
                
                # Clamp Delta to [-10, 10]
                logit_delta = max(-10.0, min(10.0, logit_delta))

                # Find target nodes
                target_node_ids = self.nodes_by_axis.get(target_axis, [])
                
                for t_nid in target_node_ids:
                    if t_nid not in logits: continue
                    t_logits = logits[t_nid]
                    
                    if target_cat in t_logits:
                        old_l = t_logits[target_cat]
                        t_logits[target_cat] += logit_delta
                        
                        self.trace["applied_nudges"].append({
                            "source": key,
                            "target": f"{target_axis}:{target_cat}",
                            "nudge": logit_delta,
                            "coupling": coupling
                        })

        # Step 3: Convert back to probabilities and normalize
        new_posteriors = {}
        for nid, l_dist in logits.items():
            probs = {k: _sigmoid(v) for k, v in l_dist.items()}
            new_posteriors[nid] = _normalize(probs)
            
        return new_posteriors

    # --------- Public API ---------

    def run(self, evidence_by_node: Dict[str, Dict]) -> (Dict, Dict, Dict):
        """
        Returns (local_posteriors, final_posteriors, trace)
        """
        post = {}

        # (A) Prior * Evidence (Logit Fusion)
        for node in self.graph:
            nid = node["id"]
            if not node.get("categories"):
                post[nid] = {}
                continue
            
            prior = self._node_prior(node)
            ev = evidence_by_node.get(nid, {}).get("category_scores", {})
            
            # Ensure support alignment
            ev_aligned = {c: ev.get(c, 0.0) for c in prior.keys()}
            
            post[nid] = self._combine_prior_evidence(prior, ev_aligned)

        # (B) Apply Correlations (Logit Nudges)
        local_posteriors = {k: v.copy() for k, v in post.items()} # Snapshot before logic
        final_posteriors = self._apply_correlations(post)
        
        # Add to trace
        self.trace["local_posteriors"] = local_posteriors
        self.trace["final_posteriors"] = final_posteriors

        return local_posteriors, final_posteriors, self.trace


