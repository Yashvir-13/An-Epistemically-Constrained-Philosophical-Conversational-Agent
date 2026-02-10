# heirarchial_splitting.py
"""
DEFINES THE SCHOPENHAUER ONTOLOGY.

This file acts as the 'Source of Truth' for the philosophical taxonomy.
It defines:
1. `philosophical_concepts`: A hierarchical mapping of concepts (Suicide, Will, etc.).
2. `AXIS_SPEC`: The allowed categories for each Axis.
3. `map_concept_to_axis_category`: Helper to convert raw concepts into Axis/Category pairs.
"""

import json
import ollama
# import re  <- No longer needed
from langchain_core.documents import Document
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FILE = "structured_documents.json"
OUTPUT_FILE = "enriched_documents.json"

##############################################
# ✅ 1. SCHOPENHAUER ONTOLOGY (80+ concepts)
# 
# (UNCHANGED)
##############################################

philosophical_concepts = {
    "Metaphysics of the Will": [
        "will_as_thing_in_itself",
        "will_as_noumenal_reality",
        "world_as_representation",
        "principium_individuationis",
        "grades_of_objectification",
        "intelligible_character",
        "empirical_character"
    ],
    "Suffering & Pessimism": [
        "life_as_suffering",
        "desire_as_suffering",
        "striving_without_end",
        "boredom_as_suffering",
        "pessimism_strong",
        "meaninglessness_of_striving"
    ],
    "Compassion & Ascetic Ethics": [
        "compassion_as_basis_of_morality",
        "universalized_self",
        "justice_as_negative_compassion",
        "asceticism_and_will_negation",
        "denial_of_the_will",
        "renunciation",
        "self_overcoming"
    ],
    "Suicide & Liberation": [
        "suicide_not_morally_wrong",
        "suicide_not_denial_of_will",
        "suicide_affirms_will",
        "suicide_not_liberating",
        "suffering_as_path_to_insight",
        "denial_of_the_will_only_liberation"
    ],
    "Aesthetics": [
        "aesthetic_contemplation_as_will_less",
        "genius_as_will_free_subject",
        "sublime_as_will_suspension",
        "music_as_direct_objectification_of_will",
        "tragedy_as_insight_into_suffering"
    ],
    "Religion & Mysticism": [
        "affinity_with_buddhism",
        "affinity_with_hinduism",
        "critique_of_christian_moralism",
        "religion_as_mythical_expression_of_truth"
    ],
    "Psychology": [
        "primacy_of_unconscious_motivation",
        "irrationality_of_will",
        "self_deception",
        "empathy_as_metaphysical_recognition"
    ]
}

##############################################
# ✅ 2. Axes & categories (new, generic)
# 
# (UNCHANGED - Not currently used, but safe to keep)
##############################################

AXES_SPEC = {
    "metaphysical_status": ["affirms_will", "denies_will", "will_less", "neutral"],
    "ethical_status": ["morally_wrong", "morally_neutral", "outside_morality", "ascetic_good"],
    "psychological_cause": ["suffering", "desire", "ego", "compassion", "ignorance", "boredom"],
    "liberation_status": ["liberating", "futile", "deluded", "obscuring"],
    "aesthetic_effect": ["will_less_contemplation", "partial_suspension", "no_suspension"],
    "religious_alignment": ["buddhism_affinity", "hinduism_affinity", "christian_moralism_critique", "mythic_expression"]
}

##############################################
# ✅ 3. Robust JSON extractor (unchanged)
#
# (REMOVED - No longer necessary)
##############################################
# def extract_json_from_string(text: str): ...


##############################################
# ✅ 4. Map concept → (axis, category)
#
# (UNCHANGED - Not currently used, but safe to keep)
##############################################
def map_concept_to_axis_category(concept: str):
    c = (concept or "").lower()

    # Metaphysical / Will
    if "denial_of_the_will" in c or "will_negation" in c:
        return "metaphysical_status", "denies_will"
    if "affirms_will" in c:
        return "metaphysical_status", "affirms_will"
    if "will_less" in c or "will_free" in c or "suspension" in c:
        return "metaphysical_status", "will_less"

    # Suicide bundle
    if c.startswith("suicide_"):
        if "not_morally_wrong" in c:
            return "ethical_status", "outside_morality"
        if "not_denial_of_will" in c:
            return "metaphysical_status", "affirms_will"
        if "affirms_will" in c:
            return "metaphysical_status", "affirms_will"
        if "not_liberating" in c:
            return "liberation_status", "futile"

    # Ethics / Asceticism / Compassion
    if "ascetic" in c or "renunciation" in c or "self_overcoming" in c:
        return "ethical_status", "ascetic_good"
    if "compassion" in c or "empathy" in c or "justice_as_negative_compassion" in c:
        return "ethical_status", "morally_neutral"  # Schopenhauer's 'moral' ≠ duty; we treat as non-dutist positive neutrality

    # Suffering & Pessimism → psychological + metaphysical coloring
    if "suffering" in c or "pessim" in c or "striving" in c or "meaninglessness" in c:
        # Map primarily to psychological cause
        if "boredom" in c:
            return "psychological_cause", "boredom"
        if "desire" in c:
            return "psychological_cause", "desire"
        return "psychological_cause", "suffering"

    # Aesthetics
    if "aesthetic" in c or "sublime" in c or "genius" in c or "music" in c or "tragedy" in c:
        if "sublime" in c or "aesthetic_contemplation_as_will_less" in c or "genius" in c:
            return "aesthetic_effect", "will_less_contemplation"
        return "aesthetic_effect", "partial_suspension"

    # Religion & Mysticism
    if "buddhism" in c:
        return "religious_alignment", "buddhism_affinity"
    if "hinduism" in c:
        return "religious_alignment", "hinduism_affinity"
    if "christian_moralism" in c or "critique_of_christian_moralism" in c:
        return "religious_alignment", "christian_moralism_critique"
    if "religion_as_mythical_expression" in c or "myth" in c:
        return "religious_alignment", "mythic_expression"

    # Metaphysics general
    if "will_" in c or "world_as_representation" in c or "principium_individuationis" in c:
        return "metaphysical_status", "neutral"

    # Psychology general
    if "unconscious" in c or "irrationality" in c or "self_deception" in c:
        return "psychological_cause", "ignorance"

    # Default fallback
    return None, None

##############################################
# ✅ 5. Schopenhauer-specific enrichment (CORRECTED)
##############################################
def enrich_document_with_llm(doc: Document) -> Document:
    # Flatten ontology into a list
    all_concepts = [c for group in philosophical_concepts.values() for c in group]

    # Generate concept list for prompt
    concept_lines = "\n".join([f'      "{c}": 0.0,' for c in all_concepts])
    concept_lines = concept_lines.rstrip(',')
    
    # --- PROMPT UPDATED ---
    # Made the JSON-only instruction more explicit.
    prompt = f"""
You are an expert philosopher specializing in Arthur Schopenhauer.
Analyze the following text and determine how strongly (from -1.0 to 1.0) it expresses or rejects each Schopenhauerian concept.

Text:
{doc.page_content[:4000]}

Output *ONLY* the JSON object defined below. Do not add any other text, preamble, or explanation.
Update the 0.0 values with your scores. Do not add new keys. Do not alter the JSON structure.

{{ "topic": "<best topic among these: {list(philosophical_concepts.keys())}>", "belief_state": {{ {concept_lines} }} }}
"""

    try:
        # --- FIX APPLIED ---
        # 1. Added format="json" to force valid JSON output from llama3.
        # 2. This makes the regex parser (extract_json_from_string) unnecessary.
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            format="json",  # <-- THE KEY FIX
            options={"temperature": 0.0}
        )
        
        # 3. Directly parse the response content, which is now guaranteed to be a JSON string.
        analysis = json.loads(response["message"]["content"])

        if analysis:
            doc.metadata['topic'] = analysis.get('topic', 'Unknown')
            doc.metadata['belief_state'] = analysis.get('belief_state', {})
        else:
            raise ValueError("LLM returned empty or null JSON.")

    except Exception as e:
        # Added a more descriptive error log to help debug
        doc_id = doc.metadata.get('source', 'Unknown Doc') # Assumes 'source' is in metadata
        print(f"⚠️ LLM_Error processing doc ({doc_id}): {e}")
        doc.metadata['topic'] = 'LLM_Error'
        doc.metadata['belief_state'] = {}

    return doc

##############################################
# ✅ 6. Main Execution (unchanged)
##############################################
if __name__ == "__main__":
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        base_docs = json.load(f)

    structural_docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in base_docs
    ]

    enriched_docs = []

    # ✅ Adjust this based on your hardware (start small: 2–4)
    MAX_WORKERS = 8

    print(f"🧠 Using {MAX_WORKERS} parallel threads for enrichment...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(enrich_document_with_llm, doc): doc for doc in structural_docs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Enriching with Schopenhauer Ontology"):
            try:
                enriched_doc = future.result()
                enriched_docs.append(enriched_doc)
            except Exception as e:
                print(f"⚠️ Error enriching doc: {e}")

    # ✅ Save output
    output = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in enriched_docs
    ]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✅ Enrichment complete.")