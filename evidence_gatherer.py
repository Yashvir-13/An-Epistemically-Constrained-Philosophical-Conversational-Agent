# evidence_gatherer.py  — Philosopher AI v2.2 (axis- & subject-aware retrieval)
# GENERAL, QUERY-INDEPENDENT FIXES
import concurrent.futures
import json
import re
import math
from typing import List, Dict, Tuple


import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from Heirarchial_Splitting import (
    philosophical_concepts,
    map_concept_to_axis_category,
    AXES_SPEC,
)

LLM_MODEL = "llama3"
VECTORSTORE_DIR = "schopenhauer_vector_db"
EMBEDDING_MODEL = "intfloat/e5-large-v2"

TOP_K_DOCUMENTS = 16       # retrieve more initially
FINAL_K_DOCUMENTS = 6      # keep only best after re-ranking

# ---- Embedding model for light semantic ops ----

# ----- Flatten ontology concepts -----
ALL_CONCEPTS: List[str] = []
for group in philosophical_concepts.values():
    ALL_CONCEPTS.extend(group)
ALL_CONCEPTS_LOWER = [c.lower() for c in ALL_CONCEPTS]


# ---------- Utilities ----------

def _extract_json(txt: str):
    m = re.search(r'\{.*\}', txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        return None
    
def _process_node(self, node: Dict) -> Tuple[str, Dict]:
        """Helper function for parallel execution."""
        try:
            node_id = node["id"]
            axis = node.get("axis", "")
            categories = node.get("categories", [])
            question = node.get("question", "")
            query = node.get("search_query", question)

            # 1. Gather docs for this node
            docs = self._get_axis_docs(axis, query)

            # 2. Score docs with LLM for this node
            result = self._axis_score_llm(axis, categories, question, docs)
            
            return node_id, result
        except Exception as e:
            print(f"Error processing node {node.get('id')}: {e}")
            return node.get("id"), {}
def parse_belief_state(meta_value):
    """
    Robustly parse the belief_state field, handling:
      - dicts (already parsed)
      - JSON strings
      - double-encoded JSON strings
      - None or invalid values
    """
    if isinstance(meta_value, dict):
        return meta_value
    if not meta_value:
        return {}
    if isinstance(meta_value, str):
        try:
            # First decode once
            data = json.loads(meta_value)
            # If it’s still a stringified JSON, decode again
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}



def _word_tokens(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-]+", s.lower())


def extract_subject_tokens(query: str) -> List[str]:
    """
    Heuristic: find tokens that prefix-match any ontology concept name.
    This generalizes beyond 'suicide' (e.g., 'asceticism', 'compassion', 'tragedy').
    """
    toks = _word_tokens(query)
    if not toks:
        return []
    subjects = set()
    for t in toks:
        for c in ALL_CONCEPTS_LOWER:
            # match start-of-concept (e.g., 'suicide_' or 'asceticism')
            if c.startswith(t) or t in c.split("_"):
                subjects.add(t)
    # keep short, stable list
    return sorted(list(subjects))[:3]


# ---------- Axis-aware query expansion ----------

AXIS_HINTS = {
    "metaphysical_status": [
        "will to live", "affirmation of will", "denial of will",
        "will-less", "representation"
    ],
    "ethical_status": [
        "morality", "compassion", "asceticism", "virtue", "justice", "duty"
    ],
    "psychological_cause": [
        "motivation", "inner state", "suffering", "desire", "ego", "character"
    ],
    "liberation_status": [
        "liberation", "deliverance", "ascetic will-denial", "freedom from will"
    ],
    "aesthetic_effect": [
        "aesthetic contemplation", "sublime", "will suspension", "genius", "art"
    ],
    "religious_alignment": [
        "Buddhism", "Hinduism", "Christianity", "mysticism", "myth"
    ],
}

def expand_query(query: str, axis: str, subjects: List[str]) -> str:
    hints = " ".join(AXIS_HINTS.get(axis, []))
    subj_str = " ".join(subjects) if subjects else ""
    return f"{query} {hints} {subj_str}".strip()


# ---------- Axis relevance & conflation control ----------

def score_doc_axis_relevance(doc, axis: str) -> float:
    """
    Score how much the doc's belief_state touches the target axis.
    """
    meta = doc.metadata or {}
    beliefs = parse_belief_state(meta.get("belief_state"))


    axis_score, axis_total = 0.0, 0.0
    for concept, val in beliefs.items():
        ax, _ = map_concept_to_axis_category(concept)
        if ax == axis:
            axis_score += max(0.0, val + 1.0)  # (-1..1) -> (0..2)
        axis_total += 1.0

    if axis_total == 0.0:
        return 0.0
    return axis_score / axis_total


def subject_alignment_score(doc, subjects: List[str]) -> float:
    """
    Reward docs whose belief_state concept names contain the subject tokens.
    Generalizes to any subject appearing in ontology (not hard-coded).
    """
    if not subjects:
        return 0.0

    meta = doc.metadata or {}
    beliefs = parse_belief_state(meta.get("belief_state"))


    hits = 0.0
    total = 0.0
    for concept in beliefs.keys():
        c = concept.lower()
        total += 1.0
        if any((s in c.split("_")) or c.startswith(s) for s in subjects):
            hits += 1.0

    if total == 0.0:
        return 0.0
    return hits / total


def conflation_penalty(doc, subjects: List[str]) -> float:
    """
    Penalize docs heavily dominated by non-subject phenomena when the query
    clearly names a subject that exists in our ontology (e.g., asceticism vs suicide).
    Works for ANY subject token found by extract_subject_tokens().
    """
    if not subjects:
        return 1.0  # no-op if no subject detected

    meta = doc.metadata or {}
    beliefs = parse_belief_state(meta.get("belief_state"))


    subject_weight = 0.0
    nonsubj_weight = 0.0

    for concept, val in beliefs.items():
        c = concept.lower()
        w = abs(float(val))
        if any((s in c.split("_")) or c.startswith(s) for s in subjects):
            subject_weight += w
        else:
            nonsubj_weight += w

    # If non-subject dominates a lot, apply penalty
    if subject_weight == 0.0 and nonsubj_weight > 0.0:
        return 1.0 / (1.0 + nonsubj_weight)
    ratio = subject_weight / (subject_weight + nonsubj_weight + 1e-6)
    # map ratio (0..1) -> penalty (0.5 .. 1.0)
    return 0.5 + 0.5 * ratio


def cross_axis_penalty(doc, target_axis: str) -> float:
    """
    Softly penalize documents whose belief_state is dominated by other axes.
    """
    meta = doc.metadata or {}
    beliefs = meta.get("belief_state", {})
    if isinstance(beliefs, str):
        try:
            beliefs = json.loads(beliefs)
        except:
            beliefs = {}

    other = 0.0
    for concept, val in beliefs.items():
        ax, _ = map_concept_to_axis_category(concept)
        if ax and ax != target_axis:
            other += abs(float(val))

    return 1.0 / (1.0 + other)  # more other-axis weight → smaller factor


def total_relevance(doc, axis: str, subjects: List[str]) -> float:
    """
    Final re-ranking score combining:
      - axis relevance
      - subject alignment
      - anti-conflation
      - cross-axis penalty
    """
    a = score_doc_axis_relevance(doc, axis)             # 0..(2-normalized)
    s = subject_alignment_score(doc, subjects)          # 0..1
    cp = conflation_penalty(doc, subjects)              # 0.5..1
    xp = cross_axis_penalty(doc, axis)                  # (0,1]

    # weights chosen empirically; adjust if needed
    return (0.45 * a) + (0.30 * s) + (0.15 * cp) + (0.10 * xp)


# ---------- Axis-specific LLM guidance ----------

AXIS_GUIDE = {
    "metaphysical_status": "Focus ONLY on will, denial/affirmation of will, will-less states, representation.",
    "ethical_status": "Focus ONLY on compassion, asceticism, wrongdoing, justice, and 'outside morality' framing.",
    "psychological_cause": "Focus ONLY on suffering, desire, ego, character, motives, inner condition.",
    "liberation_status": "Focus ONLY on liberation vs futility vs delusion vs obscuring with respect to will.",
    "aesthetic_effect": "Focus ONLY on art, sublime, aesthetic contemplation, will-suspension.",
    "religious_alignment": "Focus ONLY on Buddhism, Hinduism, Christianity, mysticism, mythic expression.",
}

SYSTEM_HINT = """
Schopenhauer interprets phenomena along distinct axes; do not mix axes.
Classify ONLY on the requested axis. Do not substitute adjacent phenomena.
If the query names a phenomenon (e.g., 'suicide', 'asceticism'), classify THAT phenomenon,
not a different but related one (e.g., do not conflate ascetic starvation with suicide).
"""


# ---------- EvidenceGatherer ----------

class EvidenceGatherer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"}
        )
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K_DOCUMENTS})

    def _get_axis_docs(self, axis: str, query: str) -> List:
        subjects = extract_subject_tokens(query)
        expanded = expand_query(query, axis, subjects)
        docs = self.retriever.invoke(expanded)

        # --- Fix: Deserialize belief_state once after retrieval ---
        for doc in docs:
            if 'belief_state' in doc.metadata:
                parsed = parse_belief_state(doc.metadata['belief_state'])
                doc.metadata['belief_state'] = parsed
        # ----------------------------------------------------------

        
        scored = [(doc, total_relevance(doc, axis, subjects)) for doc in docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = [x[0] for x in scored[:FINAL_K_DOCUMENTS]]
        return top_docs


    def _axis_score_llm(self, axis: str, categories: List[str], question: str, docs: List) -> Dict:
        if not docs:
             return {
                "core_claim": "No relevant documents found.",
                "category_scores": {c: 1.0/len(categories) for c in categories},
                "confidence": 0.0,
                "context": ""
            }

        # --- CRITICAL FIX: Keyword Guardrailing ---
        # If the question is about a specific subject (e.g., "suicide"), 
        # and that subject does NOT appear in the retrieved text, 
        # force the confidence to 0 to prevent hallucination.
        
        q_subjects = extract_subject_tokens(question) # e.g., ['suicide']
        combined_text = " ".join([d.page_content.lower() for d in docs])
        
        # Check if at least one subject token exists in the text
        subject_present = False
        if not q_subjects:
            subject_present = True # General question
        else:
            for s in q_subjects:
                if s in combined_text:
                    subject_present = True
                    break
        
        if not subject_present:
            print(f"   [Filter] Subject '{q_subjects}' not found in retrieval. Aborting LLM scoring.")
            return {
                "core_claim": f"The retrieved text discusses {docs[0].metadata.get('topic', 'other topics')} but does not explicitly mention {q_subjects}.",
                "category_scores": {c: 1.0/len(categories) for c in categories}, # Uniform dist
                "confidence": 0.0, # Zero confidence
                "context": docs[0].page_content[:800]
            }
        # ------------------------------------------

        context = "\n\n---\n\n".join([d.page_content[:1200] for d in docs])
        guide = AXIS_GUIDE.get(axis, "")
        phenomenon = ", ".join(q_subjects) if q_subjects else "the named phenomenon"

        # Added explicit Schopenhauer constraints to the prompt
        prompt = f"""
You are an expert reasoning engine for a Schopenhauer simulation.
Analyze the TEXT to classify the PHENOMENON according to Schopenhauer's specific philosophy.

AXIS: {axis} 
GUIDE: {guide} 
CATEGORIES: {categories} 
PHENOMENON: {phenomenon}
QUESTION: "{question}"

TEXT: {context}

INSTRUCTIONS:
1. If the TEXT does not explicitly support an answer, return "answerable": false.
2. DO NOT use outside knowledge. Base the answer ONLY on the TEXT.
3. If the text is about a different topic (e.g., asceticism generally) but not specifically {phenomenon}, return "answerable": false.

Return ONLY JSON: {{ "answerable": <bool>, "core_claim": "<concise summary>", "category_scores": {{ "<category>": 0..1 }}, "confidence": 0..1 }}
"""

        try:
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                keep_alive=-1
            )
            j = _extract_json(resp["message"]["content"]) or {}

        except Exception:
            j = {}
        
        if not j.get("answerable", True):
            # Fallback to uniform
            u = 1.0 / len(categories) if categories else 0
            return {
                "core_claim": "Context insufficient.",
                "category_scores": {c: u for c in categories},
                "confidence": 0.1, 
                "context": docs[0].page_content[:800]
            }

        scores = j.get("category_scores", {})
        if scores and isinstance(scores, dict):
            # Normalize
            s = sum(float(v) for v in scores.values()) or 1.0
            scores = {k: float(v) / s for k, v in scores.items() if k in categories}
        else:
            if categories:
                u = 1.0 / len(categories)
                scores = {c: u for c in categories}

        return {
            "core_claim": j.get("core_claim", ""),
            "category_scores": scores,
            "confidence": float(j.get("confidence", 0.6)),
            "context": docs[0].page_content[:800]
        }

        scores = j.get("category_scores", {})
        if scores and isinstance(scores, dict):
            s = sum(float(v) for v in scores.values()) or 1.0
            scores = {k: float(v) / s for k, v in scores.items() if k in categories}
        else:
            # uniform fallback
            if categories:
                u = 1.0 / len(categories)
                scores = {c: u for c in categories}

        return {
            "core_claim": j.get("core_claim", ""),
            "category_scores": scores,
            "confidence": float(j.get("confidence", 0.6)),
            "context": docs[0].page_content[:800] if docs else ""
        }
    def _process_node(self, node: Dict) -> Tuple[str, Dict]:
        """Helper function for parallel execution."""
        try:
            node_id = node["id"]
            axis = node.get("axis", "")
            categories = node.get("categories", [])
            question = node.get("question", "")
            query = node.get("search_query", question)

            # 1. Gather docs for this node
            print(f"[Thread] Processing node: {node_id} ({axis})")
            docs = self._get_axis_docs(axis, query)

            # 2. Score docs with LLM for this node
            result = self._axis_score_llm(axis, categories, question, docs)
            
            print(f"[Thread] Completed node: {node_id}")
            return node_id, result
        except Exception as e:
            print(f"[Thread] Error processing node {node.get('id')}: {e}")
            return node.get("id"), {}

    def gather_evidence_for_graph(self, graph: List[Dict]) -> Dict[str, Dict]:
        results = {}
        
        # We only need to process nodes that have categories (i.e., not 'final_conclusion')
        nodes_to_process = [node for node in graph if node.get("categories")]
        
        if not nodes_to_process:
             # Handle graphs with no processing nodes (like the fallback)
             for node in graph:
                results[node["id"]] = {}
             return results

        # Use a ThreadPoolExecutor to run _process_node in parallel
        # Set max_workers to a reasonable number, e.g., 5, or len(nodes_to_process)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes_to_process)) as executor:
            # Submit all tasks
            future_to_node = {
                executor.submit(self._process_node, node): node
                for node in nodes_to_process
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_node):
                node_id, result = future.result()
                if result:
                    results[node_id] = result
                else:
                    # Ensure entry exists even if processing failed
                    results[node_id] = {} 

        # Ensure all nodes have an entry, even if empty (for 'final_conclusion')
        for node in graph:
            if node["id"] not in results:
                results[node["id"]] = {}

        return results