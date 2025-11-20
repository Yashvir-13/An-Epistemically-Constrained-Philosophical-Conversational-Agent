import json
import ollama
from tqdm import tqdm

INPUT_FILE = "benchmark_results_v2.json"
REPORT_FILE = "evaluation_report_v2.md"
LLM_MODEL = "llama3" # Or use "gpt-4" if you have API keys, but llama3 works for coarse grading

def grade_response(question, golden_points, system_answer, q_type):
    
    golden_str = "\n- ".join(golden_points)
    
    prompt = f"""
    You are an impartial academic judge evaluating an AI simulating Arthur Schopenhauer.
    
    QUESTION: {question}
    
    REQUIRED FACTS (The answer MUST align with these):
    - {golden_str}
    
    AI ANSWER: 
    {system_answer}
    
    EVALUATION TASK:
    1. Does the AI answer align with the Required Facts?
    2. If this is a TRICK question (adversarial), did the AI correctly reject the false premise?
    3. Did the AI hallucinate religious/moralizing content (e.g. "sin") that Schopenhauer would reject?
    
    SCORE (1-5):
    1 = Completely wrong, hallucinated, or fell for the trick.
    3 = Mostly correct but vague or missed a key nuance.
    5 = Perfect philosophical accuracy and consistency.
    
    Return ONLY JSON: {{ "score": <int>, "reasoning": "<short explanation>" }}
    """
    
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], format='json', options={'temperature': 0.0})
        return json.loads(response['message']['content'])
    except:
        return {"score": 1, "reasoning": "Judge LLM failed to parse output."}

def run_evaluation():
    with open(INPUT_FILE, 'r') as f:
        results = json.load(f)
        
    total_score = 0
    trick_score = 0
    trick_count = 0
    valid_score = 0
    valid_count = 0
    
    detailed_log = []
    
    print("⚖️ Judging responses...")
    for res in tqdm(results):
        grade = grade_response(res['question'], res['golden_answer_points'], res['system_answer'], res['type'])
        
        res['grade'] = grade
        detailed_log.append(res)
        
        score = grade['score']
        total_score += score
        
        if res['type'] == 'adversarial':
            trick_score += score
            trick_count += 1
        else:
            valid_score += score
            valid_count += 1

    # Calculate Averages
    avg_total = total_score / len(results) if results else 0
    avg_valid = valid_score / valid_count if valid_count else 0
    avg_trick = trick_score / trick_count if trick_count else 0
    
    report = f"""
# SchopenhauerBench Evaluation Report

## Summary Metrics
- **Overall Faithfulness Score:** {avg_total:.2f} / 5.0
- **Standard Questions Score:** {avg_valid:.2f} / 5.0
- **Adversarial/Trick Score:** {avg_trick:.2f} / 5.0  (Did it hallucinate?)

## Interpretation
- **> 4.0**: Research Grade. Publishable performance.
- **3.0 - 4.0**: Good prototype. Needs ontology tuning.
- **< 3.0**: System is hallucinating or retrieving poorly.

## Failure Modes (Low Scores)
"""
    
    for res in detailed_log:
        if res['grade']['score'] <= 2:
            report += f"\n### Q: {res['question']}\n"
            report += f"- **Type:** {res['type']}\n"
            report += f"- **Agent Answer:** {res['system_answer'][:200]}...\n"
            report += f"- **Judge Reasoning:** {res['grade']['reasoning']}\n"

    with open(REPORT_FILE, 'w') as f:
        f.write(report)
        
    print(f"✅ Evaluation Complete. Report saved to {REPORT_FILE}")
    print(f"   Overall Score: {avg_total:.2f}/5")

if __name__ == "__main__":
    run_evaluation()