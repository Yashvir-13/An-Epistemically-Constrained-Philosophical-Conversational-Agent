# probabilistic_reasoner.py — Philosopher AI v2.2
# Multi-axis categorical inference with cross-axis consistency

import math
from typing import Dict, List


def _softmax_map(logits: Dict[str, float]) -> Dict[str, float]:
    if not logits:
        return {}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in logits.keys()}


class ProbabilisticReasoner:
    """
    - priors_by_axis[axis][category] = prob
    - evidence_by_node[node_id]["category_scores"] = prob vector (same support as node categories)
    - Cross-axis consistency nudges:
        * metaphysical_status:
            - affirms_will ↔ liberation_status: futile
            - denies_will  ↔ liberation_status: liberating
            - will_less    ↔ liberation_status: liberating (weaker)
        * ethical_status:
            - ascetic_good ↔ metaphysical_status: denies_will
            - outside_morality ↔ metaphysical_status: affirms_will (soft)
    """

    def __init__(self, graph: List[Dict], priors_by_axis: Dict[str, Dict[str, float]],
                 evidence_weight: float = 1.0, consistency_strength: float = 0.4):
        self.graph = graph
        self.priors_by_axis = priors_by_axis or {}
        self.evidence_weight = float(evidence_weight)
        self.consistency_strength = float(consistency_strength)

        # index nodes by axis
        self.nodes_by_axis = {}
        for n in graph:
            ax = n.get("axis", "")
            if ax:
                self.nodes_by_axis.setdefault(ax, []).append(n["id"])

        # quick lookup
        self.node_by_id = {n["id"]: n for n in graph}

    def _node_prior(self, node: Dict) -> Dict[str, float]:
        axis = node.get("axis", "")
        cats = node.get("categories", [])
        px = self.priors_by_axis.get(axis, {})
        if not cats:
            return {}
        total = sum(px.get(c, 0.0) for c in cats)
        if total <= 0.0:
            u = 1.0 / len(cats)
            return {c: u for c in cats}
        return {c: px.get(c, 0.0) / total for c in cats}

    def _combine_prior_evidence(self, prior: Dict[str, float], evidence: Dict[str, float]) -> Dict[str, float]:
        # log-add weighting
        logits = {}
        for c in prior.keys():
            p = max(1e-12, prior.get(c, 1e-12))
            e = max(1e-12, evidence.get(c, 1e-12))
            logits[c] = math.log(p) + self.evidence_weight * math.log(e)
        return _softmax_map(logits)

    # --------- Cross-axis consistency nudges (general, not query-specific) ---------

    def _consistency_nudges(self, posteriors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        
        # Helper to get top cat
        def map_cat(nid):
            d = posteriors.get(nid, {})
            if not d: return None
            return max(d.items(), key=lambda kv: kv[1])[0]

        # Find node ids
        meta_nodes = self.nodes_by_axis.get("metaphysical_status", [])
        ethic_nodes= self.nodes_by_axis.get("ethical_status", [])
        lib_nodes  = self.nodes_by_axis.get("liberation_status", [])

        # 1. Metaphysical (Master Axis) -> drives Ethical and Liberation
        for mid in meta_nodes:
            m_best = map_cat(mid)
            if not m_best: continue

            # Nudge Ethical based on Metaphysical
            for eid in ethic_nodes:
                edist = dict(posteriors.get(eid, {}))
                if not edist: continue
                bump = {c: 0.0 for c in edist.keys()}
                
                if m_best == "affirms_will":
                    # If it affirms will (suicide), it is NOT ascetic good, and usually outside standard morality
                    if "outside_morality" in bump: bump["outside_morality"] += self.consistency_strength * 1.5
                    if "morally_neutral" in bump: bump["morally_neutral"] += self.consistency_strength
                    # Strongly penalize "morally_wrong" if it implies sin
                    if "morally_wrong" in bump: bump["morally_wrong"] -= self.consistency_strength 

                elif m_best == "denies_will":
                    if "ascetic_good" in bump: bump["ascetic_good"] += self.consistency_strength * 1.5

                # Apply
                logits = {c: math.log(max(edist.get(c, 1e-9), 1e-9)) + bump[c] for c in edist}
                posteriors[eid] = _softmax_map(logits)

            # Nudge Liberation based on Metaphysical
            for lid in lib_nodes:
                ldist = dict(posteriors.get(lid, {}))
                if not ldist: continue
                bump = {c: 0.0 for c in ldist.keys()}

                if m_best == "affirms_will":
                    if "futile" in bump: bump["futile"] += self.consistency_strength * 1.5
                elif m_best == "denies_will":
                    if "liberating" in bump: bump["liberating"] += self.consistency_strength
                
                logits = {c: math.log(max(ldist.get(c, 1e-9), 1e-9)) + bump[c] for c in ldist}
                posteriors[lid] = _softmax_map(logits)

        return posteriors

    # --------- Public API ---------

    def run(self, evidence_by_node: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """
        Returns posteriors: node_id -> {category: prob}
        """
        post = {}

        # (A) prior × evidence per node
        for node in self.graph:
            nid = node["id"]
            if not node.get("categories"):
                post[nid] = {}
                continue
            prior = self._node_prior(node)
            ev = evidence_by_node.get(nid, {}).get("category_scores", {})
            # ensure same support
            ev = {c: ev.get(c, 1e-9) for c in prior.keys()}
            post[nid] = self._combine_prior_evidence(prior, ev)

        # (B) cross-axis consistency nudges
        post = self._consistency_nudges(post)

        return post
