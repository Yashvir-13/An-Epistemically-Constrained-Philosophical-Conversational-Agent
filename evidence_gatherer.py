import concurrent.futures
import json
import re
import math
from typing import List, Dict, Tuple

import ollama
import numpy as np
from numpy.linalg import norm
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

TOP_K_DOCUMENTS = 20       # retrieve more initially
FINAL_K_DOCUMENTS = 10      # keep only best after re-ranking

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

def parse_belief_state(meta_value):
    """Robustly parse the belief_state field."""
    if isinstance(meta_value, dict):
        return meta_value
    if not meta_value:
        return {}
    if isinstance(meta_value, str):
        try:
            data = json.loads(meta_value)
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

# ---------- Axis-aware relevance scoring ----------

def score_doc_axis_relevance(doc, axis: str) -> float:
    meta = doc.metadata or {}
    beliefs = parse_belief_state(meta.get("belief_state"))
    axis_score, axis_total = 0.0, 0.0
    for concept, val in beliefs.items():
        ax, _ = map_concept_to_axis_category(concept)
        if ax == axis:
            axis_score += max(0.0, val + 1.0)
        axis_total += 1.0
    if axis_total == 0.0: return 0.0
    return axis_score / axis_total

def subject_alignment_score(doc, subjects: List[str]) -> float:
    if not subjects: return 0.0
    meta = doc.metadata or {}
    beliefs = parse_belief_state(meta.get("belief_state"))
    hits, total = 0.0, 0.0
    for concept in beliefs.keys():
        c = concept.lower()
        total += 1.0
        # Check if subject matches concept key loosely
        if any((s in c) for s in subjects):
            hits += 1.0
    if total == 0.0: return 0.0
    return hits / total

def total_relevance(doc, axis: str, subjects: List[str]) -> float:
    a = score_doc_axis_relevance(doc, axis)
    s = subject_alignment_score(doc, subjects)
    # Simple weighted sum
    return (0.6 * a) + (0.4 * s)

# ---------- Axis-specific LLM guidance ----------

AXIS_GUIDE = {
    "metaphysical_status": "Focus ONLY on will, denial/affirmation of will, will-less states, representation.",
    "ethical_status": "Focus ONLY on compassion, asceticism, wrongdoing, justice, and 'outside morality' framing.",
    "psychological_cause": "Focus ONLY on suffering, desire, ego, character, motives, inner condition.",
    "liberation_status": "Focus ONLY on liberation vs futility vs delusion vs obscuring with respect to will.",
    "aesthetic_effect": "Focus ONLY on art, sublime, aesthetic contemplation, will-suspension.",
    "religious_alignment": "Focus ONLY on Buddhism, Hinduism, Christianity, mysticism, mythic expression.",
}

class EvidenceGatherer:
    """
    Handles Retrieval Augmented Generation (RAG) for the Schopenhauer agent.
    
    Pipeline:
    1. **Vector Search**: Queries ChromaDB for Schopenhauer's texts using semantic embeddings.
    2. **Concept Matching**: Semantically extracts 'Subjects' (e.g. 'suicide', 'music') from the query.
    3. **Re-ranking**: Scores documents based on (Vector Similarity + Axis Relevance + Subject Match).
    4. **Context Scoring**: Uses LLM to check if the retrieved text ACTUALLY answers the question.
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"} # Change to 'cpu' if needed
        )
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K_DOCUMENTS})
        
        # Pre-compute concept embeddings
        self.concept_vectors = {}
        vectors = self.embeddings.embed_documents(ALL_CONCEPTS_LOWER)
        for concept, vec in zip(ALL_CONCEPTS_LOWER, vectors):
            self.concept_vectors[concept] = vec

    def extract_subjects_semantic(self, query: str, threshold: float = 0.70) -> List[str]:
        """
        Extracts ontology concepts that match the query semantically.
        
        Args:
            query (str): User question.
            threshold (float): Similarity threshold (default 0.70).
            
        Returns:
            list: Top 3 matching concepts (e.g. ['will_to_live', 'suffering']).
        """
        query_vec = self.embeddings.embed_query(query)
        matches = []
        for concept, vec in self.concept_vectors.items():
            sim = np.dot(query_vec, vec) / (norm(query_vec) * norm(vec))
            if sim > threshold:
                matches.append(concept)
        return matches[:3] # Top 3 concepts

    def _get_axis_docs(self, axis: str, query: str) -> List:
        # 1. Expand query with axis hints
        hints = " ".join(AXIS_GUIDE.get(axis, "").split()[:10]) # Use first few words of guide
        expanded = f"{query} {hints}"
        
        docs = self.retriever.invoke(expanded)

        # 2. Deserialize belief_state
        for doc in docs:
            if 'belief_state' in doc.metadata:
                doc.metadata['belief_state'] = parse_belief_state(doc.metadata['belief_state'])

        # 3. Re-rank
        subjects = self.extract_subjects_semantic(query)
        scored = [(doc, total_relevance(doc, axis, subjects)) for doc in docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [x[0] for x in scored[:FINAL_K_DOCUMENTS]]

    def get_doc_context(self, axis: str, query: str) -> str:
        """
        Retrieves raw document context without LLM scoring.
        Used for building the final context window for synthesis.
        """
        try:
            docs = self._get_axis_docs(axis, query)
            if not docs: return ""
            return "\n\n---\n\n".join([d.page_content[:1000] for d in docs])
        except Exception as e:
            print(f"Error fetching docs for {axis}: {e}")
            return ""

    def _axis_score_llm(self, axis: str, categories: List[str], question: str, docs: List) -> Dict:
        if not docs:
             return {
                "core_claim": "No relevant documents found.",
                "category_scores": {c: 1.0/len(categories) for c in categories},
                "confidence": 0.0,
                "context": ""
            }

        context = "\n\n---\n\n".join([d.page_content[:1000] for d in docs])
        guide = AXIS_GUIDE.get(axis, "")

        prompt = f"""
You are an expert reasoning engine for a Schopenhauer simulation.
Analyze the TEXT to classify the PHENOMENON according to Schopenhauer's philosophy.

AXIS: {axis} 
GUIDE: {guide} 
CATEGORIES: {categories} 
QUESTION: "{question}"

TEXT: {context}

INSTRUCTIONS:
1. If the TEXT is irrelevant to the question, return "answerable": false.
2. DO NOT use outside knowledge. Base the answer ONLY on the TEXT.

Return ONLY JSON: {{ "answerable": <bool>, "core_claim": "<concise summary>", "category_scores": {{ "<category>": 0..1 }}, "confidence": 0..1 }}
"""

        try:
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0},
                keep_alive=-1
            )
            j = json.loads(resp["message"]["content"])
        except Exception:
            j = {}
        
        if not j.get("answerable", True):
            u = 1.0 / len(categories) if categories else 0
            return {
                "core_claim": "Context insufficient.",
                "category_scores": {c: u for c in categories},
                "confidence": 0.1, 
                "context": docs[0].page_content[:500]
            }

        scores = j.get("category_scores", {})
        # Normalize scores
        if scores and isinstance(scores, dict):
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

    def _process_node(self, node: Dict) -> Tuple[str, Dict]:
        try:
            node_id = node["id"]
            axis = node.get("axis", "")
            categories = node.get("categories", [])
            question = node.get("question", "")
            query = node.get("search_query", question)

            # print(f"[Thread] Processing node: {node_id} ({axis})")
            docs = self._get_axis_docs(axis, query)
            result = self._axis_score_llm(axis, categories, question, docs)
            # print(f"[Thread] Completed node: {node_id}")
            
            return node_id, result
        except Exception as e:
            print(f"[Thread] Error processing node {node.get('id')}: {e}")
            return node.get("id"), {}

    def gather_evidence_for_graph(self, graph: List[Dict]) -> Dict[str, Dict]:
        results = {}
        nodes_to_process = [node for node in graph if node.get("categories")]
        
        if not nodes_to_process:
             for node in graph: results[node["id"]] = {}
             return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes_to_process)) as executor:
            future_to_node = {executor.submit(self._process_node, node): node for node in nodes_to_process}
            for future in concurrent.futures.as_completed(future_to_node):
                node_id, result = future.result()
                if result: results[node_id] = result
                else: results[node_id] = {} 

        for node in graph:
            if node["id"] not in results: results[node["id"]] = {}

        return results