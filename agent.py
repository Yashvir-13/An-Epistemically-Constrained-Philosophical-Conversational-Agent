"""
agent.py — Philosopher AI v2.2 (Schopenhauer)
Main orchestration script connecting:
  - reasoning_graph generator
  - evidence gatherer (axis & subject aware)
  - probabilistic reasoner (multi-axis)
  - synthesis module (philosophical persona)
"""

import json
import ollama
import traceback
from reasoning_graph import generate_reasoning_graph
from evidence_gatherer import EvidenceGatherer
from probabilistic_reasoner import ProbabilisticReasoner
from Initial_priors import build_global_priors
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------
# LLM model for synthesis
# --------------------------
SYNTHESIS_MODEL = "llama3"


# --------------------------
# Utility for clean printing
# --------------------------

def pretty_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


# --------------------------
# Schopenhauer synthesis prompt
# --------------------------

def synthesize_schopenhauer_answer(graph, evidence, posteriors, user_question):
    """
    Compose final answer using the philosopher's voice, integrating across axes.
    """
    # Condense axis-level summaries
    axis_summary = []
    for node in graph:
        if not node.get("categories"):
            continue
        nid = node["id"]
        axis = node.get("axis")
        
        # Get the MATHEMATICALLY determined top category
        dist = posteriors.get(nid, {})
        topcat = max(dist.items(), key=lambda kv: kv[1])[0] if dist else "N/A"
        
        # Only include evidence text if confidence was high
        ev = evidence.get(nid, {})
        conf = ev.get("confidence", 0)
        core = ev.get("core_claim", "") if conf > 0.4 else "Evidence weak; relying on priors."
        
        axis_summary.append(
            f"- Axis '{axis}': \n  * Determinted Category: {topcat} (Prob: {dist.get(topcat,0):.2f})\n  * Textual Context: {core}"
        )
    summary_str = "\n".join(axis_summary)

    # --- THE GUARDRAILS ---
    schopenhauer_axioms = """
    CORE AXIOMS (DO NOT VIOLATE):
    1. The Will-to-Live is the thing-in-itself; it is blind, irrational striving.
    2. Life is essentially suffering (pendulum between pain and boredom).
    3. Morality is based on Compassion, NOT duty.
    4. SUICIDE: It is a mistake/affirmation of will, but NOT a moral crime/sin.
    
    BIOGRAPHICAL TRUTHS (DO NOT HALLUCINATE):
    - I never married. I lived a solitary life.
    - I despised G.W.F. Hegel. I called him a charlatan. I did NOT support him.
    - I did not live to see the full Industrial Revolution; I focus on timeless suffering, not specific industrial praise.
    """

    prompt = f"""
    You are simulating Arthur Schopenhauer. 

    User Question: "{user_question}"

    {schopenhauer_axioms}

    ANALYTIC FINDINGS FROM YOUR BRAIN (Use these):
    {summary_str}

    CRITICAL INSTRUCTION ON TRUTH:
    - The user may ask "Trick Questions" with false premises (e.g., "Why did you love Hegel?").
    - If the user's premise contradicts your Biography or Philosophy, **YOU MUST REJECT THE PREMISE**.
    - Do NOT "play along" or invent reasons for things that never happened.
    - Instead, correct the user in a grumpy, superior tone (e.g., "Absurd! I never did such a thing...").

    Example Rejection:
    User: "Why did you marry?"
    Answer: "Your question is born of ignorance. I never entangled myself in the trap of marriage. I remained solitary to better contemplate the Will."

    Final Answer (in Schopenhauer's voice):
    """

    try:
        response = ollama.chat(
            model=SYNTHESIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}, # Keep temp low for consistency
            keep_alive=-1
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print("[Error in synthesis]", e)
        return "The Will obstructs my thought process."
# --------------------------
# Main interactive loop
# --------------------------

def main():
    print("Initializing Philosopher AI v2 (Schopenhauer)...")
    print("\n--- Philosopher AI v2 ---")
    print("Type 'exit' to quit.\n")

    gatherer = EvidenceGatherer()

    while True:
        try:
            question = input("> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            # 1️⃣ Build reasoning graph
            graph = generate_reasoning_graph(question)
            print("\n[Graph]")
            print(pretty_json(graph))

            # 2️⃣ Gather evidence (axis-aware retrieval + scoring)
            evidence = gatherer.gather_evidence_for_graph(graph)
            print("\n[Evidence]")
            print(pretty_json(evidence))

            # 3️⃣ Compute priors for axes
            priors = build_global_priors()

            # 4️⃣ Probabilistic reasoning (categorical Bayesian inference)
            reasoner = ProbabilisticReasoner(graph, priors)
            posteriors = reasoner.run(evidence)
            print("\n[Posteriors]")
            print(pretty_json(posteriors))

            # 5️⃣ Philosophical synthesis
            answer = synthesize_schopenhauer_answer(graph, evidence, posteriors, question)
            print("\n--- Schopenhauer's Answer ---\n")
            print(answer)
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print("\n[ERROR]")
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()
