
import json
import ollama
import re
import time
from tqdm import tqdm

INPUT_FILE = "benchmarks/raw_evaluation_results.json"
OUTPUT_FILE = "benchmarks/fidelity_report.json"
MARKDOWN_REPORT = "benchmarks/fidelity_report.md"
JUDGE_MODEL = "gemma2"

def clean_json(text):
    """
    Strips markdown code blocks from JSON output.
    """
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def blind_judge(question, answer_a, answer_b):
    """
    Uses Gemma 2 to blindly and comparatively evaluate philosophical fidelity to Schopenhauer.
    
    It asks the Judge (Gemma 2) to rate:
    1. Ontological Accuracy (Will vs Representation)
    2. Conceptual Precision (Correct vocabulary)
    3. Metaphysical Discipline (Avoiding psychologism)
    4. Tone Fidelity (Pessimism, Arrogance)
    
    Returns a detailed JSON structure with the winner and reasoning.
    """

    prompt = f"""
You are a strict and uncompromising scholar of Arthur Schopenhauer.
You are evaluating TWO anonymous answers to the SAME philosophical question.

Your task is COMPARATIVE, not absolute.
You must judge which answer is MORE faithful to Schopenhauer's philosophy.

Schopenhauerian Evaluation Dimensions:
1. Ontological Accuracy
   - Correct distinction between Will (thing-in-itself) and Representation
   - Avoids conflating suspension of will with denial of will

2. Conceptual Precision
   - Uses Schopenhauerian concepts correctly (e.g., aesthetic contemplation, ascetic denial, Ideas)
   - Avoids category errors or conceptual smoothing

3. Metaphysical Discipline
   - Does NOT over-extend metaphysics where only phenomenology applies
   - Does NOT psychologize metaphysical or aesthetic states

4. Tone Fidelity
   - Appropriately pessimistic, severe, unsentimental
   - Avoids optimism, consolation, or modern therapeutic framing

5. Overall Philosophical Fidelity
   - Which answer a Schopenhauer scholar would endorse as more correct

IMPORTANT RULES:
- Do NOT reward eloquence or verbosity.
- Penalize confident but incorrect assertions.
- Penalize answers that collapse aesthetic, ethical, and metaphysical domains.
- If both answers are equally good or equally flawed, you may declare a Tie.

QUESTION:
"{question}"

ANSWER A:
"{answer_a}"

ANSWER B:
"{answer_b}"

Your Output MUST be valid JSON and ONLY JSON.

Return exactly this structure:

{{
  "ontology": "A" | "B" | "Tie",
  "conceptual_precision": "A" | "B" | "Tie",
  "metaphysical_discipline": "A" | "B" | "Tie",
  "tone_fidelity": "A" | "B" | "Tie",
  "overall_winner": "A" | "B" | "Tie",
  "confidence": "low" | "medium" | "high",
  "category_error": {{
    "A": true | false,
    "B": true | false,
    "description": "Brief description if any category error is present, otherwise empty string"
  }},
  "reason": "Concise scholarly justification (2–4 sentences max)"
}}
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = ollama.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0}
            )

            content = res["message"]["content"]
            cleaned = clean_json(content)
            return json.loads(cleaned)

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "ontology": "Tie",
                    "conceptual_precision": "Tie",
                    "metaphysical_discipline": "Tie",
                    "tone_fidelity": "Tie",
                    "overall_winner": "Tie",
                    "confidence": "low",
                    "category_error": {
                        "A": False,
                        "B": False,
                        "description": f"Judge error after {max_retries} retries: {str(e)}"
                    },
                    "reason": "Judging failed due to repeated parsing or model errors."
                }
            time.sleep(1)
            # Backoff slightly

def run_fidelity_check():
    """
    Runs the 'Philosophical Fidelity Benchmark' (External Persona Test).
    
    This evaluates "How much like Schopenhauer does it sound?" compared to a Baseline (Vanilla Llama 3).
    It uses A/B testing with an LLM Judge.
    
    Outputs:
    - JSON results: `benchmarks/fidelity_report.json`
    - Markdown Report: `benchmarks/fidelity_report.md`
    """
    print(f"⚖️  Measuring Fidelity (Judge: {JUDGE_MODEL}) - RE-RUN (Judge Only)...")
    
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    # Flatten questions from groups
    all_items = []
    for group in data:
        all_items.extend(group["responses"])

    # RESUME LOGIC
    processed_questions = set()
    results = {
        "judge_votes": {"Fathom": 0, "Baseline": 0, "Tie": 0},
        "details": []
    }

    try:
        with open(OUTPUT_FILE, "r") as f:
            existing_results = json.load(f)
            # Validate schema matches what we expect
            if "details" in existing_results and isinstance(existing_results["details"], list):
                results = existing_results
                for d in results["details"]:
                    processed_questions.add(d["question"])
                print(f"🔄 Resuming... Found {len(processed_questions)} already judged questions.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("🆕 Starting fresh run.")
        
    for item in tqdm(all_items):
        question = item["fathom"]["question"]
        
        if question in processed_questions:
            continue

        fathom_ans = item["fathom"]["answer"]
        baseline_ans = item["baseline"]["answer"]
        
        # 2. Blind Judge
        judge_res = blind_judge(question, fathom_ans, baseline_ans)
        winner_code = judge_res.get("overall_winner", "Tie")
        
        winner_name = "Tie"
        if winner_code == "A":
            winner_name = "Fathom"
        elif winner_code == "B":
            winner_name = "Baseline"
            
        results["judge_votes"][winner_name] += 1
        
        results["details"].append({
            "question": question,
            "judge_winner": winner_name,
            "judge_reason": judge_res.get("reason", ""),
            "metrics": {
                "ontology": judge_res.get("ontology", "Tie"),
                "conceptual_precision": judge_res.get("conceptual_precision", "Tie"),
                "metaphysical_discipline": judge_res.get("metaphysical_discipline", "Tie"),
                "tone_fidelity": judge_res.get("tone_fidelity", "Tie"),
                "confidence": judge_res.get("confidence", "low")
            }
        })

        # INCREMENTAL SAVE
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # --- GENERATE DETAILED REPORT ---
    total = len(results["details"])
    if total == 0:
        print("No results to report.")
        return

    # Calculate sub-metric stats
    metrics_stats = {
        "ontology": {"Fathom": 0, "Baseline": 0, "Tie": 0},
        "conceptual_precision": {"Fathom": 0, "Baseline": 0, "Tie": 0},
        "metaphysical_discipline": {"Fathom": 0, "Baseline": 0, "Tie": 0},
        "tone_fidelity": {"Fathom": 0, "Baseline": 0, "Tie": 0},
        "overall_winner": {"Fathom": 0, "Baseline": 0, "Tie": 0} # Re-calculate to safely include resumed data
    }

    # Aggregate
    for d in results["details"]:
        # Overall
        w = d["judge_winner"] # Already "Fathom"/"Baseline"/"Tie"
        metrics_stats["overall_winner"][w] += 1
        
        # Sub-metrics
        m = d.get("metrics", {})
        for key in ["ontology", "conceptual_precision", "metaphysical_discipline", "tone_fidelity"]:
            val_code = m.get(key, "Tie")
            val_name = "Tie"
            if val_code == "A": val_name = "Fathom"
            elif val_code == "B": val_name = "Baseline"
            # If sub-metric is missing or weird, count as Tie
            
            metrics_stats[key][val_name] += 1

    md_report = f"""# 🧠 Philosophical Fidelity Report (Judge: {JUDGE_MODEL})

## 🏆 Summary
- **Total Questions Evaluated**: {total}
- **Fathom Wins**: {metrics_stats['overall_winner']['Fathom']} ({metrics_stats['overall_winner']['Fathom']/total:.0%})
- **Baseline Wins**: {metrics_stats['overall_winner']['Baseline']} ({metrics_stats['overall_winner']['Baseline']/total:.0%})
- **Ties**: {metrics_stats['overall_winner']['Tie']}

## 📊 Detailed Metrics Breakdown
| Metric | 🌊 Fathom (A) | 🤖 Baseline (B) | 🤝 Tie | Fathom Dominance |
| :--- | :---: | :---: | :---: | :---: |
| **Ontological Accuracy** | {metrics_stats['ontology']['Fathom']} | {metrics_stats['ontology']['Baseline']} | {metrics_stats['ontology']['Tie']} | {metrics_stats['ontology']['Fathom']/total:.0%} |
| **Conceptual Precision** | {metrics_stats['conceptual_precision']['Fathom']} | {metrics_stats['conceptual_precision']['Baseline']} | {metrics_stats['conceptual_precision']['Tie']} | {metrics_stats['conceptual_precision']['Fathom']/total:.0%} |
| **Metaphysical Discipline** | {metrics_stats['metaphysical_discipline']['Fathom']} | {metrics_stats['metaphysical_discipline']['Baseline']} | {metrics_stats['metaphysical_discipline']['Tie']} | {metrics_stats['metaphysical_discipline']['Fathom']/total:.0%} |
| **Tone Fidelity** | {metrics_stats['tone_fidelity']['Fathom']} | {metrics_stats['tone_fidelity']['Baseline']} | {metrics_stats['tone_fidelity']['Tie']} | {metrics_stats['tone_fidelity']['Fathom']/total:.0%} |
| **OVERALL VERDICT** | **{metrics_stats['overall_winner']['Fathom']}** | **{metrics_stats['overall_winner']['Baseline']}** | **{metrics_stats['overall_winner']['Tie']}** | **{metrics_stats['overall_winner']['Fathom']/total:.0%}** |

## 💡 Qualitative Sample
"""
    for d in results["details"][:5]:
        md_report += f"- **Q**: {d['question']}\n  - **Winner**: {d['judge_winner']}\n  - **Reason**: {d['judge_reason']}\n"

    with open(MARKDOWN_REPORT, "w") as f:
        f.write(md_report)
        
    print(md_report)

if __name__ == "__main__":
    run_fidelity_check()
