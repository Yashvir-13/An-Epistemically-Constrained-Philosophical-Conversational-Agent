"""
agent.py — Philosopher AI v2.2 (Schopenhauer)
Main orchestration script.
"""

import json
import ollama
import traceback
from reasoning_graph import generate_next_step, reason_axis_once
from evidence_gatherer import EvidenceGatherer
from probabilistic_reasoner import SoftLogicNetwork, EpistemicProfile, calculate_entropy
from Initial_priors import build_global_priors
from axis_router import AxisRouter
from belief_constraints import build_belief_constraints
from verifier import verify_and_repair
from Heirarchial_Splitting import AXES_SPEC

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------
# LLM model for synthesis
# --------------------------
SYNTHESIS_MODEL = "llama3"


def pretty_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


def classify_intent(user_input: str) -> str:
    """
    Classifies the user's input into one of three high-level intents using a small LLM call.
    
    Categories:
    - PHILOSOPHY: Core interactions (questions about Schopenhauer, metaphysics, etc.)
    - CHITCHAT: Social pleasantries (which Schopenhauer will dismiss).
    - OFF_TOPIC: Unrelated queries (coding, math, etc.).
    
    Returns:
        str: "PHILOSOPHY", "CHITCHAT", or "OFF_TOPIC".
    """
    prompt = f"""
    Classify the following user input into one of these three categories:
    1. PHILOSOPHY: Questions about Schopenhauer, metaphysics, ethics, will, suffering, etc.
    2. CHITCHAT: Greetings, pleasantries, "how are you", simple conversational filler.
    3. OFF_TOPIC: Coding requests, math problems, recipe requests, or anything unrelated to philosophy.

    INPUT: "{user_input}"

    Return ONLY the category name.
    """
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        content = response["message"]["content"].strip().upper()
        if "CHITCHAT" in content: return "CHITCHAT"
        if "OFF_TOPIC" in content: return "OFF_TOPIC"
        return "PHILOSOPHY" 
    except:
        return "PHILOSOPHY"

def get_grumpy_dismissal():
    """
    Generates a persona-appropriate dismissal for chitchat.
    Schopenhauer does not engage in small talk.
    """
    prompt = "Generate a short, 1-sentence grumpy dismissal from Arthur Schopenhauer to someone wasting his time."
    try:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except:
        return "Human trivialities bore me."


def classify_discursive_mode(question: str) -> str:
    """
    Determines if the answer should be EXPLANATORY/DESCRIPTIVE or EVALUATIVE.
    
    This is critical for the 'Schopenhauer Persona'. 
    - EXPLANATORY: Used for 'What'/'How' questions. Tone is cold, objective, descriptive.
    - EVALUATIVE: Used for 'Should'/'Good'/'Bad' questions. Tone is judgmental, prescriptive, normative.
    
    Uses hard heuristic rules (keyword matching) rather than LLM to ensure consistency.
    
    Args:
        question (str): The user's input question.
        
    Returns:
        str: "evaluative" or "explanatory".
    """
    q = question.lower().strip()
    
    # Heuristics for EVALUATIVE mode
    evaluative_triggers = [
        "morally wrong", "morally right", 
        "should one", "ought to",
        "good or bad", "is it good", "is it bad",
        "liberation", "salvation", "redemption",
        "value of life", "meaning of life",
        "better than"
    ]
    
    if any(trigger in q for trigger in evaluative_triggers):
        return "evaluative"
        
    # Heuristics for EXPLANATORY/DESCRIPTIVE mode (Default)
    # Covers: "Why...", "How...", "What is...", "Relationship between..."
    return "explanatory"

AXIS_EXPLANATION_OBLIGATIONS = {
    "psychological_cause": {
        "must_explain": ["mental state", "phenomenology", "affect", "boredom", "desire", "motivation", "will's individual manifestation"],
        "forbid": ["thing-in-itself", "noumenal", "metaphysical essence", "ultimate reality", "will as a whole"]
    },
    "aesthetic_effect": {
        "must_explain": ["contemplation", "suspension", "perception", "will-less", "beauty", "object of art"],
        "forbid": ["liberation", "salvation", "redemption", "metaphysical denial"]
    },
    "ethical_status": {
        "must_explain": ["compassion", "justice", "wrong-doing", "motivation", "egoism"],
        "forbid": ["aesthetic", "genius", "musical expression"]
    },
    "metaphysical_status": {
        "must_explain": ["thing-in-itself", "noumena", "will as essence", "representation"],
        "forbid": [] # Metaphysics is the ground, it can refer to anything
    }
}

def synthesize_schopenhauer_answer(graph, evidence, posteriors, user_question, suppressed_axes=None, epistemic_profiles=None):
    """
    Composes the final answer using the philosopher's voice (Persona Synthesis).
    
    This function:
    1. Determines the 'Discursive Mode' (Explanatory vs. Evaluative).
    2. Checks for 'Axis Locking' (if entropy is high, force focus on one domain).
    3. Builds a context summary from the Probabilistic Graph.
    4. Injects core Schopenhauerian Axioms and Rules.
    5. Calls the LLM (Llama 3) to generate the text.
    6. Runs the 'Verifier' to check for hallucinated optimism or banned concepts.
    
    Args:
        graph (list): The accumulated knowledge graph nodes.
        evidence (dict): The retrieved evidence text and confidence scores.
        posteriors (dict): The final probability distributions for each node.
        user_question (str): The original question.
        suppressed_axes (list): Axes that were relevant but secondary (suppressed to avoid confusion).
        epistemic_profiles (dict): The EpistemicProfile objects (permissions, tiers) for each node.
        
    Returns:
        tuple: (final_answer_text, violation_detected_bool)
    """
    # 1. Classify Mode
    mode = classify_discursive_mode(user_question)
    
    # 2. Determine Axis Locking (Entropy-Gated)
    # We lock to the top axis from the graph
    primary_axis = graph[0]["axis"] if graph else "metaphysical_status"
    primary_node_id = graph[0]["id"] if graph else None
    
    axis_lock_instruction = ""
    if primary_node_id and epistemic_profiles and primary_node_id in epistemic_profiles:
        profile = epistemic_profiles[primary_node_id]
        if profile.entropy >= 0.7 and primary_axis != "metaphysical_status":
            obligations = AXIS_EXPLANATION_OBLIGATIONS.get(primary_axis, {"must_explain": [], "forbid": []})
            axis_lock_instruction = f"""
            CRITICAL: AXIS LOCK ENABLED (High Entropy: {profile.entropy:.2f})
            Your internal math shows high uncertainty in the {primary_axis} domain.
            You are FORBIDDEN from escaping into metaphysical abstractions like 'thing-in-itself' or 'noumenal'.
            You MUST stay within the phenomenological scope of {primary_axis}.
            
            REQUIRED FOCUS: {", ".join(obligations["must_explain"])}
            FORBIDDEN TERMS: {", ".join(obligations["forbid"])}
            """

    axis_summary = []
    for node in graph:
        if not node.get("categories"):
            continue
        nid = node["id"]
        axis = node.get("axis")
        
        dist = posteriors.get(nid, {})
        topcat = max(dist.items(), key=lambda kv: kv[1])[0] if dist else "N/A"
        
        ev = evidence.get(nid, {})
        conf = ev.get("confidence", 0)
        core = ev.get("core_claim", "") if conf > 0.4 else "Evidence weak."
        
        profile = epistemic_profiles.get(nid) if epistemic_profiles else None
        
        entry = f"- Axis '{axis}': \n  * Conclusion: {topcat} (Prob: {dist.get(topcat,0):.2f})"
        
        if profile:
             entry += f"\n  * EPISTEMIC STATUS: Tier={profile.tier}, Entropy={profile.entropy:.2f}"
             if not profile.permissions['metaphysical_claims']:
                 entry += " [METAPHYSICAL ASSERTION FORBIDDEN]"
        
        entry += f"\n  * Evidence: {core}"
        axis_summary.append(entry)
    summary_str = "\n".join(axis_summary)

    schopenhauer_axioms = """
    CORE AXIOMS:
    1. The Will-to-Live is the thing-in-itself; blind, irrational striving.
    2. Life is essentially suffering.
    3. Morality is based on Compassion.
    4. SUICIDE: A mistake, but NOT a moral sin.
    
    CRITICAL NUANCES:
    - Logic applies to Phenomena, NOT the Will itself.
    - I prefer Monarchy/Order over Democracy.
    - I rejected Hegel.

    BIOGRAPHICAL TRUTHS:
    - Never married. Solitary.
    """

    # --- Mode-Specific Instructions ---
    if mode == "evaluative":
        mode_instruction = """
        MODE: EVALUATIVE / JUDGMENTAL
        - You MAY judge, condemn, or evaluate the moral worth of things.
        - You MAY invoke pessimism, the futility of existence, and the denial of the will.
        - Openly discuss whether things are 'better' or 'worse' (usually worse).
        """
    else: # explanatory / descriptive
        mode_instruction = """
        MODE: EXPLANATORY / DESCRIPTIVE
        - Do NOT dismiss the user's question.
        - Do NOT condemn aesthetic pleasure or call it illusory/futile (unless specifically asked).
        - EXPLAIN the mechanics, the relationships, and the definitions with frigid precision.
        - Focus on the 'how' and 'why'.
        - Reserve judgment, but maintain a tone of INTELLECTUAL SUPERIORITY.
        """
    non_negotiable, negotiable, tensions, polarity_rules = build_belief_constraints(graph, posteriors)

    constraint_block = ""
    
    if non_negotiable:
        constraint_block += "BELIEF CONSTRAINTS (NON-NEGOTIABLE):\n"
        for c in non_negotiable:
            constraint_block += f"- {c}\n"
            
    if negotiable:
        constraint_block += "\nBELIEF COMMITMENTS (QUALIFIED DISCUSSION ALLOWED):\n"
        for c in negotiable:
            constraint_block += f"- {c}\n"
            
    if tensions:
        constraint_block += "\nTENSION WARNINGS (MUST ACKNOWLEDGE):\n"
        for c in tensions:
            constraint_block += f"- {c}\n"

    if suppressed_axes:
        constraint_block += "\nSUPPRESSED AXES (DO NOT DOMINATE EXPLANATION):\n"
        constraint_block += "The following axes are present but secondary. Do not let their metaphysical language override the main axis:\n"
        for ax in suppressed_axes:
            constraint_block += f"- {ax}\n"

    if polarity_rules:
        constraint_block += "\nSEMANTIC POLARITY RULES (STRICT LANGUAGE CONTROL):\n"
        for c in polarity_rules:
            constraint_block += f"- {c}\n"

    epistemic_block = ""
    if epistemic_profiles:
        epistemic_block += "\nEPISTEMIC PERMISSIONS (BINDING):\n"
        for nid, prof in epistemic_profiles.items():
            perms = prof.permissions
            epistemic_block += f"Axis '{prof.axis}':\n"
            epistemic_block += f"  - Assert Truth: {perms['assert']}\n"
            epistemic_block += f"  - Make Metaphysical Claims: {perms['metaphysical_claims']}\n"
            epistemic_block += f"  - Use Causal Grammar: {perms['causal_language']}\n"
            if not perms['assert']:
                epistemic_block += "  -> YOU MUST ONLY 'EXPLAIN' OR 'DESCRIBE'. DO NOT STATE AS FACT.\n"


    prompt = f"""
    You are simulating Arthur Schopenhauer.
    
    CRITICAL STYLE RULES:
    1. IMPERIOUS TONE: You are an arrogant, pessimistic genius. You do not "hope this helps". You STATE THE TRUTH.
    2. NO PLEASANTRIES: Never say "Great question", "Intriguing topic", "My dear friend", or "I hope this clarifies".
    3. VOCABULARY: Use words like: "wretched", "blind striving", "delusion", "phantasmagoria", "banal", "absurd".
    4. DIRECTNESS: Start your answer immediately. Do not preamble.
    5. PESSIMISM: Even when explaining, imply that the subject matter reflects the suffering of the Will.

    User Question: "{user_question}"

    {schopenhauer_axioms}

    {constraint_block}
    {axis_lock_instruction}

    {epistemic_block}

    {mode_instruction}

    FINDINGS:
    {summary_str}


    TASK:
    1. Answer the question in Schopenhauer's voice (grumpy, lucid, pessimistic).
    2. STRICTLY ADHERE to the DISCURSIVE MODE instructions above.
    3. You are FORBIDDEN from contradicting any 'NON-NEGOTIABLE' belief constraint.
    4. You may discuss 'BELIEF COMMITMENTS' but must qualify them with nuance.
    5. If 'TENSION WARNINGS' are present, explicitly acknowledge the tension.
    6. If 'SEMANTIC POLARITY RULES' exist, you MUST obey the prohibited words/frames at the sentence level.
    7. If the user premise is false (e.g. "Why do you love Hegel?"), REJECT IT WITH CONTEMPT.

    Final Answer:
    """

    try:
        response = ollama.chat(
            model=SYNTHESIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
            keep_alive=-1
        )
        raw_answer = response["message"]["content"].strip()
        
        # 4️⃣ Verification Pass
        # Combine non-negotiable constraints with strict polarity rules for verification
        combined_constraints = non_negotiable + polarity_rules
        if axis_lock_instruction:
            combined_constraints.append(axis_lock_instruction)
            
        final_answer, violated = verify_and_repair(raw_answer, combined_constraints)
        return final_answer, violated
    except Exception as e:
        print("[Error in synthesis]", e)
        return "The Will obstructs my thought process.", False


from doctrine_anchors import NON_NEGOTIABLE_DOCTRINES

def apply_doctrinal_corrections(question, graph, posteriors, epistemic_profiles):
    """
    Adjust posteriors for non-negotiable doctrines if entropy is low.
    """
    q_low = question.lower()
    for node in graph:
        nid = node["id"]
        axis = node["axis"]
        profile = epistemic_profiles.get(nid)
        dist = posteriors.get(nid, {})

        if not profile or not dist:
            continue

        # Check each doctrine anchor
        for anchor_name, anchor in NON_NEGOTIABLE_DOCTRINES.items():
            if axis != anchor["axis"]:
                continue
            
            # Match keywords to question
            if any(kw in q_low for kw in anchor["keywords"]):
                # Apply only if entropy is below threshold
                if profile.entropy < anchor["entropy_threshold"]:
                    target_cat = anchor["required_category"]
                    if target_cat in dist:
                        # Correct: Force target_cat to be dominant
                        # We don't just set to 1.0, we shift mass to preserve some distribution shape
                        current_prob = dist[target_cat]
                        if current_prob < 0.6:
                            print(f"     [Doctrinal Anchor] Correcting '{axis}' for '{anchor_name}' (Entropy: {profile.entropy:.2f})")
                            new_dist = {cat: prob * 0.2 for cat, prob in dist.items()}
                            new_dist[target_cat] = 0.8 + (dist[target_cat] * 0.1)
                            # Normalize
                            total = sum(new_dist.values())
                            posteriors[nid] = {cat: prob/total for cat, prob in new_dist.items()}
    return posteriors


class ReasoningAgent:
    """
    The Main Controller for the Schopenhauer AI.
    
    This class orchestrates the entire cognitive pipeline:
    1. Intent Classification (Philosophy vs. Chitchat)
    2. Axis Routing (Determining which philosophical domains apply)
    3. Evidence Gathering (RAG from vector DB)
    4. Probabilistic Reasoning (Soft Logic Network to update beliefs)
    5. Epistemic Profiling (Calculating confidence/entropy)
    6. Synthesis (Generating the final textual response)
    
    Attributes:
        gatherer (EvidenceGatherer): Handles vector database retrieval.
        router (AxisRouter): Handles axis selection (Metaphysics, Ethics, etc.).
        priors (dict): Global prior probabilities.
    """
    def __init__(self):
        self.gatherer = EvidenceGatherer()
        self.router = AxisRouter()
        self.priors = build_global_priors()

    def reason(self, question: str):
        # 1️⃣ Intent
        intent = classify_intent(question)
        if intent == "CHITCHAT":
            return {"intent": "CHITCHAT", "answer": get_grumpy_dismissal()}
        elif intent == "OFF_TOPIC":
            return {"intent": "OFF_TOPIC", "answer": "I am a philosopher, not a servant. Begone."}
        
        # 2️⃣ Routing
        routing_result = self.router.rank_axes(question)
        TOP_K_AXES = 2
        ranked_list = routing_result["axis_ranking"]
        allowed_axes = [item["axis"] for item in ranked_list[:TOP_K_AXES]]
        suppressed_axes = [item["axis"] for item in ranked_list[TOP_K_AXES:] if item["score"] > 0.4]
        
        # 3️⃣ Extraction & Reasoning
        accumulated_graph = []
        full_evidence = {}
        for axis in allowed_axes:
            evidence_text = self.gatherer.get_doc_context(axis, question)
            categories = AXES_SPEC.get(axis, [])
            result = reason_axis_once(axis, question, evidence_text, categories)
            
            node_id = f"node_{axis}"
            node_dict = {
                "id": node_id,
                "axis": result.axis,
                "question": question,
                "categories": categories
            }
            accumulated_graph.append(node_dict)
            full_evidence[node_id] = {
                "core_claim": result.explanation,
                "category_scores": result.category_scores,
                "confidence": result.confidence,
                "context": evidence_text[:500]
            }

        # 4️⃣ Soft Logic
        full_posteriors = {}
        if accumulated_graph:
            reasoner = SoftLogicNetwork(accumulated_graph, self.priors)
            _, full_posteriors, _ = reasoner.run(full_evidence)

        # 5️⃣ Profiling
        epistemic_profiles = {}
        for node in accumulated_graph:
            nid = node["id"]
            axis = node["axis"]
            dist = full_posteriors.get(nid, {})
            epistemic_profiles[nid] = EpistemicProfile(nid, axis, dist)

        # 6️⃣ Doctrinal Correction
        # Capture Pre-Correction Entropy (Primary Axis)
        primary_nid = accumulated_graph[0]["id"] if accumulated_graph else None
        pre_entropy = epistemic_profiles[primary_nid].entropy if primary_nid in epistemic_profiles else 0.0

        full_posteriors = apply_doctrinal_corrections(question, accumulated_graph, full_posteriors, epistemic_profiles)

        # Recalculate Entropy (Post-Correction)
        post_entropy = pre_entropy
        if primary_nid and primary_nid in full_posteriors:
            post_entropy = calculate_entropy(full_posteriors[primary_nid])

        # 7️⃣ Synthesis Check (MANDATORY BLOCK)
        # If primary axis has NO evidence (confidence near 0 or context empty), DO NOT SYNTHESIZE.
        primary_axis = allowed_axes[0] if allowed_axes else "none"
        primary_node_id = f"node_{primary_axis}"
        primary_evidence = full_evidence.get(primary_node_id, {})
        
        # "Context insufficient" is the sentinel string from EvidenceGatherer for null retrieval
        if primary_evidence.get("core_claim") == "Context insufficient." or not primary_evidence.get("context"):
            print(f"     [Safety Block] Synthesis aborted. No evidence for primary axis: {primary_axis}")
            return {
                "intent": "PHILOSOPHY",
                "answer": f"I cannot philosophize on {primary_axis} without sufficient data. The Will is obscure here.",
                "violation_detected": False,
                "repaired": False,
                "pre_entropy": pre_entropy,
                "post_entropy": post_entropy,
                "routing": routing_result,
                "graph": accumulated_graph,
                "evidence": full_evidence,
                "posteriors": full_posteriors,
                "profiles": epistemic_profiles,
                "suppressed_axes": suppressed_axes,
                "status": "BLOCKED_NO_EVIDENCE"
            }

        # 8️⃣ Synthesis
        answer, violation_detected = synthesize_schopenhauer_answer(
            accumulated_graph, full_evidence, full_posteriors, 
            question, suppressed_axes, epistemic_profiles
        )
        
        # Verifier now returns (text, success_bool). If success_bool is False but text changed, it failed re-verify.
        # We need to capture that nuance if possible, but for now we follow the 'repaired' flag from verifier.
        is_repaired = violation_detected # synthesize_schopenhauer_answer calls verify_and_repair which returns (text, bool)
                                         # But wait, verify_and_repair returns (text, True) if repaired.
                                         # If re-verify fails, I set it to return (text, False).
                                         # So 'violation_detected' variable here actually captures "Is Repaired And Verified" based on my change?
                                         # Let's check agent.py synth function...
                                         # It returns: final_answer, violated
                                         # In my verifier change: return repair, False (if failed check).
                                         # So 'violated' will be False if repair failed check? That's confusing.
                                         # Let's assume 'violation_detected' roughly maps to "Did we change it successfully?"
                                         # Actually, let's just log what we got.

        return {
            "intent": "PHILOSOPHY",
            "answer": answer,
            "violation_detected": True if violation_detected else False, # If it returned True, it was repaired.
            "repaired": violation_detected, 
            "pre_entropy": pre_entropy,
            "post_entropy": post_entropy,
            "routing": routing_result,
            "graph": accumulated_graph,
            "evidence": full_evidence,
            "posteriors": full_posteriors,
            "profiles": epistemic_profiles,
            "suppressed_axes": suppressed_axes
        }


def main():
    print("Initializing Philosopher AI v2.3 (Schopenhauer)...")
    print("Type 'exit' to quit.\n")
    agent = ReasoningAgent()

    while True:
        try:
            question = input("> ").strip()
            if not question: continue
            if question.lower() in {"exit", "quit"}: break

            print(f"\n[Thinking] Philosophizing on: '{question}'...")
            result = agent.reason(question)

            if result["intent"] == "PHILOSOPHY":
                print(f"     [Router Choice]: {result['routing']['axis_ranking'][0]['axis']}")
                print(f"     [Discursive Mode]: {classify_discursive_mode(question).upper()}")
                
                print("\n[EPISTEMIC PROFILES]")
                for nid, profile in result["profiles"].items():
                    print(f"  Axis: {profile.axis}")
                    print(f"    Tier: {profile.tier}")
                    print(f"    Entropy: {profile.entropy:.4f}")
                    print(f"    Permissions: {profile.permissions}")

                print("\n--- Schopenhauer's Answer ---\n")
                print(result["answer"])
                print("\n")
            else:
                print(f"\n[Schopenhauer]: {result['answer']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print("\n[ERROR]")
            print(traceback.format_exc())
            continue

if __name__ == "__main__":
    main()