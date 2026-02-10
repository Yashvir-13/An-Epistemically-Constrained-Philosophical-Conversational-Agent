import json
import numpy as np
import math
from tqdm import tqdm
from judge_results import evaluate_single_answer, INPUT_FILE

# Use the file we just generated with debug_data
DATA_FILE = "benchmark_results_v2.json"

def calculate_kl_divergence(local_dist, final_dist):
    """
    Kullback-Leibler Divergence: D_KL(P || Q) = Sum(P(i) * log(P(i) / Q(i)))
    
    Here:
    - P (Target) = Final Posterior (After Logic)
    - Q (Source) = Local Posterior (Before Logic)
    
    Purpose:
    We measure 'Information Gain' from the Logic Layer. A high KL means the SoftLogicNetwork
    significantly changed the belief state from the raw evidence.
    """
    kl_sum = 0.0
    epsilon = 1e-9
    
    for cat, p_final in final_dist.items():
        p_local = local_dist.get(cat, epsilon)
        # Avoid log(0)
        p_final = max(p_final, epsilon)
        p_local = max(p_local, epsilon)
        
        kl_sum += p_final * math.log(p_final / p_local)
        
    return kl_sum

def calculate_brier_score(prob_dist, is_correct):
    """
    Calculates the Brier Score (Mean Squared Error of Probabilistic Prediction).
    
    Formula: (Probability_assigned_to_truth - 1.0)^2
    
    Interpretation:
    - 0.0 = Perfect Confidence aligned with Truth.
    - 1.0 = Perfect Confidence in Falsehood (Worst case).
    
    Used to measure 'Calibration'—does the agent know when it doesn't know?
    """
    if not prob_dist:
        return 0.0
        
    top_cat = max(prob_dist.items(), key=lambda x: x[1])[0]
    prob_top = prob_dist[top_cat]
    
    truth = 1.0 if is_correct else 0.0
    
    return (prob_top - truth) ** 2

def main():
    print("📊 Loading Benchmark Results...")
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
        
    total_kl = 0.0
    node_count = 0
    
    brier_scores = []
    adversarial_correct = 0
    adversarial_total = 0
    
    print(f"   Analyzing {len(data)} items...")
    
    for item in tqdm(data):
        # 1. Judge (Re-run grading for this specific run)
        grade = evaluate_single_answer(item['question'], item['golden_answer_points'], item['system_answer'], item['type'])
        
        # Determine "Correctness" based on Question Type
        if item['type'] == 'adversarial':
             # New Judge returns 1 if defended, 0 if failed
             val = grade.get('adversarial_score')
             is_correct = (val == 1)
        else:
             # Standard question: Faithfulness >= 4 is a pass
             is_correct = (grade.get('faithfulness', 0) >= 4)
        
        # 2. KL Divergence & Brier
        if 'debug_data' in item:
            local_posts = item['debug_data']['local_posteriors']
            final_posts = item['debug_data']['final_posteriors']
            
            # Iterate over all nodes in the reasoning graph for this question
            for nid, final_p in final_posts.items():
                local_p = local_posts.get(nid, {})
                
                # KL
                kl = calculate_kl_divergence(local_p, final_p)
                total_kl += kl
                node_count += 1
                
                # Brier (Only if node has meaningful distribution)
                if final_p:
                    b_score = calculate_brier_score(final_p, is_correct)
                    brier_scores.append(b_score)
        
        # 3. Adversarial Stats
        if item['type'] == 'adversarial':
            adversarial_total += 1
            if is_correct:
                adversarial_correct += 1

    # --- Summary ---
    avg_kl = total_kl / node_count if node_count > 0 else 0
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0
    adv_rejection_rate = (adversarial_correct / adversarial_total * 100) if adversarial_total > 0 else 0
    
    print("\n\n" + "="*40)
    print("      QUANTITATIVE METRICS (ACL/EMNLP)      ")
    print("="*40)
    
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Avg. KL Divergence (Info Gain) & {avg_kl:.4f} \\\\
Brier Score (Calibration error) & {avg_brier:.4f} \\\\
Adversarial Rejection Rate & {adv_rejection_rate:.1f}\% \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Quantitative performance of Neuro-Symbolic Logic}}
\\label{{tab:metrics}}
\\end{{table}}
"""
    print(latex_table)
    print("="*40)

if __name__ == "__main__":
    main()
