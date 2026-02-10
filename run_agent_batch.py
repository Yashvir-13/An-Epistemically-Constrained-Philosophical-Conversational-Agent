import json
import os
import traceback
from tqdm import tqdm

# Import your existing modules
from evidence_gatherer import EvidenceGatherer
from probabilistic_reasoner import SoftLogicNetwork, EpistemicProfile
from Initial_priors import build_global_priors
from agent import synthesize_schopenhauer_answer
from axis_router import AxisRouter


INPUT_FILE = "schopenhauer_bench.json"
OUTPUT_FILE = "benchmark_results_v2.json"

def run_batch():
    """
    Runs the full Fathom Agent logic in BATCH mode against `schopenhauer_bench.json`.
    
    Features:
    - Incremental Saving (Resumes if interrupted).
    - Top-K Axis enforcement (Matches agent.py logic).
    - Epistemic Profiling & Soft Logic updates.
    - Saves detailed debug dumps (graph, posteriors) to `benchmark_results_v2.json`.
    """
    # Load questions
    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)

    # Initialize System
    print("⚙️ Initializing Agent Components...")
    gatherer = EvidenceGatherer()
    router = AxisRouter()
    priors = build_global_priors()
    
    # Load existing results if the output file exists (for resuming)
    results = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        print(f"📂 Found existing results file. Loading to resume...")
        with open(OUTPUT_FILE, 'r') as f:
            try:
                results = json.load(f)
                processed_ids = {r['id'] for r in results}
                print(f"✓ Loaded {len(results)} existing results. Resuming from question {len(results) + 1}...")
            except json.JSONDecodeError:
                print("⚠️ Warning: Could not parse existing results file. Starting fresh.")
                results = []
                processed_ids = set()
    
     # LIMIT FOR TESTING
    total_questions = len(dataset)
    remaining = total_questions - len(processed_ids)
    print(f"🚀 Running Agent on {total_questions} questions ({remaining} remaining)")
    
    for item in tqdm(dataset, initial=len(processed_ids), total=total_questions):
        q_id = item['id']
        
        # Skip if already processed
        if q_id in processed_ids:
            continue
            
        question = item['question']
        golden = item['golden_answer_points']
        q_type = item['type']
        
        try:
            # --- ITERATIVE REASONING LOOP (Copied from agent.py) ---
            
            # --- REFACTOR: SINGLE-PASS EXECUTION (No Steps) ---
            from reasoning_graph import reason_axis_once
            from Heirarchial_Splitting import AXES_SPEC
            
            # 1. Axis Routing
            routing_result = router.rank_axes(question)
            
            # Top-K Enforcement
            TOP_K_AXES = 2
            ranked_list = routing_result["axis_ranking"]
            allowed_axes = [item["axis"] for item in ranked_list[:TOP_K_AXES]]
            
            # Identify Suppressed Axes (Ranked but not top-k)
            suppressed_axes = [item["axis"] for item in ranked_list[TOP_K_AXES:] if item["score"] > 0.4]
            
            accumulated_graph = []
            full_evidence = {}
            full_posteriors = {} 
            
            for i, axis in enumerate(allowed_axes):
                # 1. Retrieve Evidence (ONCE)
                evidence_text = gatherer.get_doc_context(axis, question)
                
                # 2. Reason (ONCE)
                categories = AXES_SPEC.get(axis, [])
                result = reason_axis_once(axis, question, evidence_text, categories)
                
                # 3. Store
                node_id = f"node_{axis}"
                
                node_dict = {
                    "id": node_id,
                    "axis": result.axis,
                    "question": question,
                    "search_query": question, 
                    "categories": categories
                }
                accumulated_graph.append(node_dict)
                
                full_evidence[node_id] = {
                    "core_claim": result.explanation,
                    "category_scores": result.category_scores,
                    "confidence": result.confidence,
                    "context": evidence_text[:500]
                }

            # 4. Global Soft Logic Update (Batch)
            if accumulated_graph:
                reasoner = SoftLogicNetwork(accumulated_graph, priors)
                _, full_posteriors, _ = reasoner.run(full_evidence)
                
            # Epistemic Profiling
            epistemic_profiles = {}
            if accumulated_graph:
                 for node in accumulated_graph:
                    nid = node["id"]
                    axis = node["axis"]
                    dist = full_posteriors.get(nid, {})
                    epistemic_profiles[nid] = EpistemicProfile(nid, axis, dist)
                
            # Variables for synthesis
            graph = accumulated_graph
            evidence = full_evidence
            final_post = full_posteriors
            local_post = {} # Not used in synthesis but kept for debug struct
            
            # 4. Synthesis
            final_answer, violated = synthesize_schopenhauer_answer(graph, evidence, final_post, question, suppressed_axes, epistemic_profiles)
            
            result_entry = {
                "id": q_id,
                "type": q_type,
                "question": question,
                "golden_answer_points": golden,
                "system_answer": final_answer,
                "violation_detected": violated,
                "graph_dump": json.dumps(graph), # Saved for debugging
                "debug_data": {
                    "local_posteriors": local_post,
                    "final_posteriors": final_post
                }
            }
            results.append(result_entry)
            
        except Exception as e:
            print(f"❌ Error on {q_id}: {e}")
            result_entry = {
                "id": q_id,
                "type": q_type,
                "question": question,
                "golden_answer_points": golden,
                "system_answer": "ERROR: System failed to generate response.",
                "error": str(e)
            }
            results.append(result_entry)
        
        # Save incrementally after each question to preserve progress
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)


    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"✅ Batch run complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_batch()
