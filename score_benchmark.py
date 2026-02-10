import json
import numpy as np
from tqdm import tqdm
from verifier import verify_and_repair
from doctrine_anchors import NON_NEGOTIABLE_DOCTRINES
import re

RAW_RESULTS = "benchmarks/raw_evaluation_results.json"
FINAL_REPORT = "benchmarks/final_report.json"

AXIS_OBLIGATIONS = {
    "psychological_cause": {
        "must": ["boredom", "desire", "suffering", "motivation", "mental"],
        "forbid": ["thing-in-itself", "noumenal", "ultimate reality"]
    },
    "aesthetic_effect": {
        "must": ["contemplation", "suspension", "perception", "beauty", "will-less"],
        "forbid": ["liberation", "salvation", "redemption"]
    },
    "ethical_status": {
        "must": ["compassion", "justice", "egoism", "wrong"],
        "forbid": ["aesthetic", "genius", "music"]
    },
    "metaphysical_status": {
        "must": ["thing-in-itself", "noumenal", "representation", "will"],
        "forbid": []
    }
}

def compute_axis_focus(answer: str, axis: str) -> float:
    if axis not in AXIS_OBLIGATIONS:
        return 1.0

    sentences = re.split(r"[.!?]+", answer.lower())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.0

    rules = AXIS_OBLIGATIONS[axis]
    aligned = 0

    for s in sentences:
        if any(f in s for f in rules["forbid"]):
            continue
        aligned += 1

    return round(aligned / len(sentences), 2)



def entropy(dist):
    p = np.array(list(dist.values()), dtype=float)
    return -np.sum(p * np.log(p + 1e-9))


def avg(xs):
    return round(float(np.mean(xs)), 2) if xs else 0.0


def compute_metrics():
    """
    Calculates aggregate metrics from the `raw_evaluation_results.json` produced by `evaluate_epistemic.py`.
    
    Metrics:
    1. **Belief Sensitivity**: Does the Posterior Probability for the correct node increase?
    2. **Doctrine Convergence**: Do Non-Negotiable Doctrines appear in the final belief state?
    3. **Entropy Resolution**: Does uncertainty decrease (Pre -> Post)?
    4. **Axis Routing Accuracy**: Did the Router pick the expected axis?
    5. **Violation Rate**: How often did the Verifier trigger?
    6. **Self-Repair Rate**: How often was a violation successfully fixed?
    
    Generates `benchmarks/final_report.json`.
    """
    with open(RAW_RESULTS) as f:
        groups = json.load(f)

    total = 0

    belief_sensitivity = []
    doctrine_convergence = []
    axis_focus = []
    entropy_resolution = []

    violations = 0
    repairs = 0
    routing_hits = 0

    baseline_focus = []
    baseline_doctrine = []
    baseline_violations = 0

    for group in groups:
        for pair in tqdm(group["responses"], desc=group["group_id"]):
            total += 1
            f = pair["fathom"]
            b = pair["baseline"]

            axis = f["target_axis"]
            expected = f["expected_direction"]
            question = f["question"].lower()

            # Routing
            if f["routing"] == axis:
                routing_hits += 1

            # ---- BELIEF SENSITIVITY ----
            node = f"node_{axis}"
            post = f.get("posteriors", {}).get(node)

            if post and expected in post:
                belief_sensitivity.append(post[expected])

            # ---- ENTROPY RESOLUTION ----
            if "pre_entropy" in f and "post_entropy" in f:
                entropy_resolution.append(f["pre_entropy"] - f["post_entropy"])

            # ---- DOCTRINE CONVERGENCE ----
            for d in NON_NEGOTIABLE_DOCTRINES.values():
                if d["axis"] == axis and any(k in question for k in d["keywords"]):
                    doctrine_convergence.append(
                        post.get(d["required_category"], 0.0) if post else 0.0
                    )
                    break

            # ---- COMPLIANCE / REPAIR ----
            if f["violation_detected"]:
                violations += 1
                if f.get("repaired", False):
                    repairs += 1

            axis_focus.append(compute_axis_focus(f["answer"], axis))

            # ---- BASELINE ----
            baseline_focus.append(compute_axis_focus(b["answer"], axis))

            _, violated = verify_and_repair(
                b["answer"],
                [f"Explain strictly in terms of {axis}"]
            )
            if violated:
                baseline_violations += 1

            if expected.replace("_", " ") in b["answer"].lower():
                baseline_doctrine.append(1.0)
            else:
                baseline_doctrine.append(0.0)

    report = {
        "belief_sensitivity": avg(belief_sensitivity),
        "doctrine_convergence": avg(doctrine_convergence),
        "axis_focus_score": avg(axis_focus),
        "entropy_resolution": avg(entropy_resolution),
        "epistemic_violation_rate": round(violations / total, 2),
        "self_repair_rate": round(repairs / max(violations, 1), 2),
        "axis_routing_accuracy": round(routing_hits / total, 2),

        "baseline_comparison": {
            "doctrine_accuracy": avg(baseline_doctrine),
            "axis_focus_score": avg(baseline_focus),
            "violation_rate": round(baseline_violations / total, 2)
        }
    }

    with open(FINAL_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {FINAL_REPORT}")


if __name__ == "__main__":
    compute_metrics()
