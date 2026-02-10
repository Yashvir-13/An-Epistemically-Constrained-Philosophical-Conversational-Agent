# ontology_miner.py
# Phase 2: Data-Driven Math (PMI Correlation Learning)

import json
import math
from collections import defaultdict
from tqdm import tqdm
from Heirarchial_Splitting import map_concept_to_axis_category

ENRICHED_DOCS_FILE = "enriched_documents.json"
OUTPUT_FILE = "learned_correlations.json"
SCORE_THRESHOLD = 0.5
MIN_COOCCURRENCE = 2

def run_mining():
    """
    Learns 'Common Sense Correlations' from the document set using Pointwise Mutual Information (PMI).
    
    If 'Suffering' appears frequently with 'Desire', this relationship is learned and saved
    to `learned_correlations.json`. These learned rules are used by the 
    SoftLogicNetwork to make probabilistic inferences across axes.
    """
    print("⛏️  Mining Schopenhauer Ontology...")
    
    try:
        with open(ENRICHED_DOCS_FILE, 'r') as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {ENRICHED_DOCS_FILE} not found.")
        return

    # 1. Count Concept Occurrences
    concept_counts = defaultdict(int)
    cooccurrence_counts = defaultdict(int) 
    doc_count = 0

    print(f"   Analyzing {len(docs)} documents...")

    for doc in tqdm(docs):
        meta = doc.get("metadata", {})
        beliefs = meta.get("belief_state", {})
        
        # Normalize belief dict if it's a string
        if isinstance(beliefs, str):
            try:
                beliefs = json.loads(beliefs)
            except:
                continue

        if not beliefs: 
            continue

        doc_count += 1
        
        # Find active concepts in this doc
        active_concepts = []
        for concept, score in beliefs.items():
            if abs(float(score)) > SCORE_THRESHOLD:
                active_concepts.append(concept)
                
        # Update counts
        for c in active_concepts:
            concept_counts[c] += 1
            
        # Update co-occurrences
        for i in range(len(active_concepts)):
            for j in range(i + 1, len(active_concepts)):
                c1 = active_concepts[i]
                c2 = active_concepts[j]
                # Store as sorted tuple to be order-agnostic
                pair = tuple(sorted((c1, c2)))
                cooccurrence_counts[pair] += 1

    # 2. Calculate PMI
    # PMI(x, y) = log( P(x, y) / (P(x) * P(y)) )
    #           = log( (count(x,y)/N) / ((count(x)/N) * (count(y)/N)) )
    #           = log( (count(x,y) * N) / (count(x) * count(y)) )
    
    correlations = {}
    
    print("   Calculating PMI scores...")
    for (c1, c2), count in cooccurrence_counts.items():
        if count < MIN_COOCCURRENCE:
            continue
            
        n = doc_count
        count_c1 = concept_counts[c1]
        count_c2 = concept_counts[c2]
        
        if count_c1 == 0 or count_c2 == 0:
            continue
            
        pmi = math.log((count * n) / (count_c1 * count_c2))
        
        # Normalize/Scale PMI to our heuristic range (-2.0 to 2.0 approx)
        # Raw PMI can go higher, but we want 'nudge' factors.
        # Let's cap and scale.
        weight = pmi * 0.5
        
        # Map concepts to Axis:Category keys
        ax1, cat1 = map_concept_to_axis_category(c1)
        ax2, cat2 = map_concept_to_axis_category(c2)
        
        if ax1 and cat1 and ax2 and cat2 and ax1 != ax2:
            # Only store cross-axis correlations
            
            # Key format: "axis:category"
            k1 = f"{ax1}:{cat1}"
            k2 = f"{ax2}:{cat2}"
            
            # Store in the structure expected by SoftLogicNetwork
            # INTER_AXIS_CORRELATION_MATRIX[k1][k2] = weight
            if k1 not in correlations: correlations[k1] = {}
            if k2 not in correlations: correlations[k2] = {}
            
            # We take the MAX correlation if multiple concept pairs map to the same category pair
            # (Simplification)
            existing_w1 = correlations[k1].get(k2, 0.0)
            if weight > existing_w1:
                correlations[k1][k2] = weight
                
            existing_w2 = correlations[k2].get(k1, 0.0)
            if weight > existing_w2:
                correlations[k2][k1] = weight

    # 3. Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(correlations, f, indent=2)
        
    print(f"✅ Learned {len(cooccurrence_counts)} pairs -> {len(correlations)} axis-correlations.")
    print(f"   Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_mining()
