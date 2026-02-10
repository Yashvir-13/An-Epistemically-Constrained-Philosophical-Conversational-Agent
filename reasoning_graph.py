# reasoning_graph.py
# Phase 3 & 4: Iterative "Tree of Thought" Reasoning + Pydantic

import ollama
import json
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional, Literal, Dict
from Heirarchial_Splitting import AXES_SPEC 

LLM_MODEL = "llama3"

# --- Pydantic Models for Single Step ---

class ReasoningNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node (e.g. 'step1_metaphysics')")
    axis: str = Field(..., description="The philosophical axis (or 'integration' if finalized)")
    question: str = Field(..., description="The specific question to investigate next")
    search_query: str = Field(..., description="Vector-optimized search query (Single string)")
    # FIX: Made categories optional (defaults to empty list) to prevent crashes
    categories: List[str] = Field(default_factory=list, description="The mutually exclusive categories to classify this into")

    @field_validator('search_query', mode='before')
    @classmethod
    def handle_list_query(cls, v):
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return v

class ReasoningStep(BaseModel):
    """
    Represents a single atomic step in the 'Tree of Thought'.
    The Agent decides to either 'CONTINUE' (explore a new axis) or be 'DONE'.
    """
    thinking_process: str = Field(..., description="Short explanation of why this step is chosen")
    node: Optional[ReasoningNode] = Field(None, description="The next node to execute. Null if status is DONE.")
    status: Literal["CONTINUE", "DONE"] = Field(..., description="Whether to continue reasoning or if we have sufficient info")

# ---------------------------------------------

def generate_next_step(user_question: str, context_so_far: List[dict], forced_axis: Optional[str] = None, allowed_axes: Optional[List[str]] = None) -> ReasoningStep:
    """
    Decides the NEXT SINGLE step in the reasoning chain using Pydantic output validation.
    
    It acts as the 'System 2' planner:
    1. Reviews the history of what has been found so far.
    2. Decides if sufficient information exists to answer.
    3. If not, generates the next optimal search query and axis/node to explore.
    """
    
    # Format context for LLM
    context_str = ""
    if not context_so_far:
        context_str = "No steps taken yet. Start with the most fundamental philosophical axis."
    else:
        context_str = json.dumps(context_so_far, indent=2)

    # --- TOP-K CONSTRAINT LOGIC ---
    if allowed_axes:
        # Only show the allowed axes to the LLM
        filtered_axes = {k: v for k, v in AXES_SPEC.items() if k in allowed_axes}
        axes_description = json.dumps(filtered_axes, indent=2)
        
        constraint_text_axes = f"""
CRITICAL CONSTRAINT (TOP-K AXES):
 You are STRICTLY LIMITED to the following axes: {json.dumps(allowed_axes)}.
 DO NOT explore any other axis. If you try, the system will reject it.
"""
    else:
        axes_description = json.dumps(AXES_SPEC, indent=2)
        constraint_text_axes = ""

    # --- FORCED AXIS LOGIC ---
    if forced_axis:
        constraint_text = f"""
CRITICAL CONSTRAINT:
You MUST start with the axis: '{forced_axis}'.
The user's question requires this specific discourse mode FIRST.
Do NOT choose any other axis for this step.
"""
    else:
        constraint_text = ""

    prompt = f"""
You are a reasoning engine for an Arthur Schopenhauer simulation.
Your goal is to answer the User Question by dynamically exploring philosophical axes ONE BY ONE.

AVAILABLE AXES: {axes_description}
{constraint_text_axes}

USER QUESTION: "{user_question}"

HISTORY OF FINDINGS:
{context_str}

TASK:
1. Analyze what we know so far.
2. Determine if we have enough information to form a complete Schopenhauerian answer.
3. If YES, set "status": "DONE".
4. If NO, create the NEXT SINGLE search node.
{constraint_text}

CRITICAL: The "node" object MUST include a list of "categories" (e.g. ["affirms_will", "denies_will"]).

OUTPUT SCHEMA (JSON):
{{
  "thinking_process": "We found X, but still need to understand Y...",
  "status": "CONTINUE" | "DONE",
  "node": {{
    "id": "step_N_axis",
    "axis": "<axis_name>",
    "question": "<specific sub-question>",
    "search_query": "<vector search query>",
    "categories": ["<cat1>", "<cat2>"] 
  }}
}}
"""
    # Retry loop to handle transient LLM formatting errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1}, # Slightly higher temp to avoid deterministic loops on error
                keep_alive=-1,
            )
            
            raw_json = response["message"]["content"]
            
            # Parse and Validate
            step_data = json.loads(raw_json)
            step = ReasoningStep(**step_data)
            return step

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"   [Reasoning Error] Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print("   [Reasoning] Max retries reached. Forcing termination.")
                return ReasoningStep(
                    thinking_process="Reasoning engine failed to generate valid JSON.",
                    status="DONE",
                    node=None
                )
        except Exception as e:
            print(f"   [System Error] {e}")
            break

    return ReasoningStep(status="DONE", thinking_process="System Error", node=None)

# Shim for legacy compatibility
def generate_reasoning_graph(question: str) -> List[dict]:
    return [
       {
            "id": "legacy_fallback",
            "axis": "metaphysical_status",
            "question": question,
            "search_query": question,
            "categories": ["affirms_will", "denies_will"]
       }
    ]

# --- SINGLE-PASS REFACTOR ---

class AxisResult(BaseModel):
    axis: str
    category: str
    confidence: float
    explanation: str
    category_scores: Dict[str, float]

def reason_axis_once(axis: str, question: str, evidence_text: str, categories: List[str]) -> AxisResult:
    """
    Performs a single-pass reasoning on a specific axis given the evidence.
    """
    if not evidence_text or len(evidence_text) < 50:
        return AxisResult(
            axis=axis,
            category="insufficient_data",
            confidence=0.0,
            explanation="No relevant evidence found.",
            category_scores={c: 0.0 for c in categories}
        )

    # Simplified Prompt
    prompt = f"""
You are an expert on Schopenhauer's philosophy.
Analyze the provided TEXT to answer the QUESTION from the perspective of the axis: '{axis}'.

CATEGORIES: {categories}

QUESTION: "{question}"

TEXT:
{evidence_text[:3000]}

TASK:
1. Determine which CATEGORY from the list above best fits the text.
2. Explain WHY (core claim).
3. Assign a confidence score (0.0 to 1.0) for the main choice.
4. Provide a probability distribution for ALL items in CATEGORIES.
   - Keys MUST match the CATEGORIES list exactly.
   - Values must sum to approx 1.0.
   - Do NOT invent new categories.

Output JSON ONLY:
{{
  "category": "<Must be one of {categories}>",
  "confidence": <float>,
  "explanation": "<concise_reasoning>",
  "category_scores": {{ "<cat1>": 0.X, "<cat2>": 0.Y ... }}
}}
"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0}
        )
        data = json.loads(response["message"]["content"])
        
        # Validate/Normalize
        scores = data.get("category_scores", {})
        
        # Filter scores to only allowed categories
        valid_scores = {k: v for k, v in scores.items() if k in categories}
        # Add missing with 0.0
        for c in categories:
            if c not in valid_scores: valid_scores[c] = 0.0
            
        best_category = data.get("category", "unknown")
        if best_category not in categories:
            # Fallback: pick highest score from VALID scores
            if valid_scores:
                best_category = max(valid_scores, key=valid_scores.get)
            else:
                best_category = categories[0] if categories else "unknown"
            
        return AxisResult(
            axis=axis,
            category=best_category,
            confidence=float(data.get("confidence", 0.0)),
            explanation=data.get("explanation", ""),
            category_scores=valid_scores
        )
    except Exception as e:
        print(f"Error in reason_axis_once: {e}")
        return AxisResult(
            axis=axis,
            category="error",
            confidence=0.0,
            explanation=f"Error: {str(e)}",
            category_scores={c: 0.0 for c in categories}
        )

if __name__ == "__main__":
    test_question = "is suicide morally permissible?"
    step = generate_next_step(test_question, [])
    print(f"--- Reasoning Step for: '{test_question}' ---")
    print(step.model_dump_json(indent=2))