import json
import os
from agent import ReasoningAgent, SYNTHESIS_MODEL
import ollama
from tqdm import tqdm

INPUT_FILE = "benchmarks/contrastive_questions.json"
OUTPUT_FILE = "benchmarks/raw_evaluation_results.json"

def run_evaluation():
    """
    Runs the 'Epistemic Truth Benchmark' (Internal Logic Test).
    
    This script tests if the Agent correctly routes and reasons about Contrastive Triples.
    Example Triple: "Why is life suffering?"
    - Expected Axis: Psychological
    - Expected Entropy: Low (Certainty)
    - Expected Direction: "Affirms Will"
    
    It runs the Agent against `benchmarks/contrastive_questions.json` and outputs 
    raw results to `benchmarks/raw_evaluation_results.json`.
    """
    if not os.path.exists("benchmarks"):
        os.makedirs("benchmarks")

    with open(INPUT_FILE, "r") as f:
        benchmarks = json.load(f)

    agent = ReasoningAgent()
    
    # LOAD EXISTING RESULTS OR INIT
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
            print(f"📂 Loaded {len(results)} groups from existing file.")
        except json.JSONDecodeError:
            print("⚠️ Warning: Could not parse existing results file. Starting fresh.")
            results = []
    else:
        results = []

    # Build a map for easy access and a set of processed IDs
    results_map = {g['group_id']: g for g in results}
    processed_ids = set()
    for g in results:
        for r in g.get('responses', []):
            if 'fathom' in r and 'id' in r['fathom']:
                processed_ids.add(r['fathom']['id'])
    
    print(f"🚀 Resuming with {len(processed_ids)} already processed questions.")
    print(f"🚀 Starting Epistemic Evaluation on {len(benchmarks)} triplets...")

    for group in benchmarks:
        group_id = group["group_id"]
        
        # Ensure group exists in results structure
        if group_id not in results_map:
            new_group_entry = {
                "group_id": group_id,
                "topic": group["topic"],
                "responses": []
            }
            results.append(new_group_entry)
            results_map[group_id] = new_group_entry
        
        group_results = results_map[group_id]
        
        # Filter questions that haven't been processed
        questions_to_run = [q for q in group["questions"] if q["id"] not in processed_ids]
        
        if not questions_to_run:
            continue

        print(f"\nProcessing Group: {group_id} ({len(questions_to_run)} items)")
        
        for q_meta in tqdm(questions_to_run, desc=f"Triplet: {group_id}"):
            question = q_meta["question"]
            
            # --- 1. Run Fathom Agent ---
            try:
                agent_res = agent.reason(question)
                
                # Extract interesting metadata
                # Note: profile.permissions is a dict, profile.entropy is a float
                extracted_profiles = {}
                for nid, prof in agent_res["profiles"].items():
                    extracted_profiles[nid] = {
                        "axis": prof.axis,
                        "tier": prof.tier,
                        "entropy": float(prof.entropy),
                        "permissions": prof.permissions
                    }

                # Extract post-synthesis violation count if possible
                # (Would require verifier to return count, but we can check if Fixed Text was used in logs if we had it)
                # For now, we'll assume the verifier logs to stdout, we might need a small patch to verifier.py 
                # to return violation status to the agent.
                
                fathom_entry = {
                    "id": q_meta["id"],
                    "question": question,
                    "target_axis": q_meta["expected_axis"],
                    "expected_direction": q_meta["expected_direction"],
                    "expected_entropy": q_meta["expected_entropy"],
                    "agent_type": "Fathom",
                    "answer": agent_res["answer"],
                    "violation_detected": agent_res["violation_detected"],
                    "repaired": agent_res.get("repaired", False),
                    "pre_entropy": agent_res.get("pre_entropy", 0.0),
                    "post_entropy": agent_res.get("post_entropy", 0.0),
                    "posteriors": agent_res["posteriors"],
                    "profiles": extracted_profiles,
                    "routing": agent_res["routing"]["axis_ranking"][0]["axis"]
                }
            except Exception as e:
                print(f"Error running agent on {q_meta['id']}: {e}")
                fathom_entry = {"id": q_meta["id"], "error": str(e)}

            # --- 2. Run Baseline (Vanilla Llama 3) ---
            try:
                # Minimal prompt for baseline to see if it over-asserts
                baseline_prompt = f"Answer this as Schopenhauer: {question}"
                baseline_resp = ollama.chat(
                    model=SYNTHESIS_MODEL,
                    messages=[{"role": "user", "content": baseline_prompt}],
                    options={"temperature": 0.3}
                )
                baseline_answer = baseline_resp["message"]["content"].strip()
                
                baseline_entry = {
                    "id": q_meta["id"],
                    "agent_type": "Baseline",
                    "answer": baseline_answer
                }
            except Exception as e:
                baseline_entry = {"id": q_meta["id"], "error": str(e)}

            # Append to memory
            group_results["responses"].append({
                "fathom": fathom_entry,
                "baseline": baseline_entry
            })

            # SAVE INCREMENTALLY
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()
