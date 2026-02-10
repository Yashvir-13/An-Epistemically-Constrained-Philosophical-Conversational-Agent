# run_baseline.py
# ABLATION STUDY: Vanilla Llama 3 (No RAG, No Reasoning Graph)

import json
import ollama
from tqdm import tqdm

INPUT_FILE = "schopenhauer_bench.json"
OUTPUT_FILE = "benchmark_results_baseline.json"
MODEL = "llama3"

SYSTEM_PROMPT = """You are an AI simulating Arthur Schopenhauer. Answer the user's question directly in his voice.
Do not use any external tools, just your internal knowledge."""

def run_baseline():
    """
    Executes the ABLATION STUDY (Control Group).
    
    It runs the unmodified 'Vanilla Llama 3' model on the same benchmark questions.
    This serves as the baseline to prove that Fathom's 'Reasoning Graph' architecture
    actually adds value (or if a simple prompt is sufficient).
    """
    try:
        with open(INPUT_FILE, 'r') as f:
            bench_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please ensure it exists.")
        return

    results = []
    print(f"Running Baseline (Vanilla {MODEL}) on {len(bench_data)} questions...")

    for item in tqdm(bench_data):
        question = item['question']
        
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                options={"temperature": 0.3}
            )
            answer = response["message"]["content"]
        except Exception as e:
            print(f"Error processing '{question}': {e}")
            answer = "Error generating response."

        # Save result in same format as v2 for easy judging
        results.append({
            "id": item.get('id', 'unknown'),
            "question": question,
            "type": item.get('type', 'standard'),
            "golden_answer_points": item.get('golden_answer_points', []),
            "system_answer": answer
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Baseline complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_baseline()
