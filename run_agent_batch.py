import json
import traceback
from tqdm import tqdm

# Import your existing modules
from reasoning_graph import generate_reasoning_graph
from evidence_gatherer import EvidenceGatherer
from probabilistic_reasoner import ProbabilisticReasoner
from Initial_priors import build_global_priors
from agent import synthesize_schopenhauer_answer

INPUT_FILE = "schopenhauer_bench.json"
OUTPUT_FILE = "benchmark_results_v2.json"

def run_batch():
    # Load questions
    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)

    # Initialize System
    print("⚙️ Initializing Agent Components...")
    gatherer = EvidenceGatherer()
    priors = build_global_priors()
    
    results = []

    print(f"🚀 Running Agent on {len(dataset)} questions...")
    
    for item in tqdm(dataset):
        q_id = item['id']
        question = item['question']
        golden = item['golden_answer_points']
        q_type = item['type']
        
        try:
            # 1. Graph
            graph = generate_reasoning_graph(question)
            
            # 2. Evidence
            evidence = gatherer.gather_evidence_for_graph(graph)
            
            # 3. Reasoning
            reasoner = ProbabilisticReasoner(graph, priors)
            posteriors = reasoner.run(evidence)
            
            # 4. Synthesis
            final_answer = synthesize_schopenhauer_answer(graph, evidence, posteriors, question)
            
            results.append({
                "id": q_id,
                "type": q_type,
                "question": question,
                "golden_answer_points": golden,
                "system_answer": final_answer,
                "graph_dump": json.dumps(graph), # Saved for debugging
                "posteriors_dump": json.dumps(posteriors) # Saved for debugging
            })
            
        except Exception as e:
            print(f"❌ Error on {q_id}: {e}")
            results.append({
                "id": q_id,
                "type": q_type,
                "question": question,
                "golden_answer_points": golden,
                "system_answer": "ERROR: System failed to generate response.",
                "error": str(e)
            })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"✅ Batch run complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_batch()
