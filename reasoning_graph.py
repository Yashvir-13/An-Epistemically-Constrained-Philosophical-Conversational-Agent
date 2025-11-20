# reasoning_graph.py

import ollama
import json
import re
from Heirarchial_Splitting import AXES_SPEC # Your new import

LLM_MODEL = "llama3"

def _extract_json_from_text(text: str):
    """Extracts the first valid JSON list or object from a text block."""
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Warning: Found JSON-like block, but failed to parse: {json_string}")
            return None
    return None

def generate_reasoning_graph(question: str) -> list[dict]:
    """
    Dynamically generates a reasoning graph with *targeted* search queries.
    """
    
    # Create the list of available axes for the prompt
    axes_description = json.dumps(AXES_SPEC, indent=2)

    prompt = f"""
You are a reasoning engine specializing in Schopenhauer's philosophy.
Your task is to deconstruct a user's question into a causal reasoning graph.
The graph must be a list of nodes (Python dictionaries).

Each node *must* have:
1.  "id": A short, unique identifier (e.g., "q1_will").
2.  "axis": The philosophical axis of analysis, chosen from the available AXES.
3.  "question": The specific sub-question that needs to be answered by an analyst.
4.  "search_query": A search query *optimized for a vector database* to find text to answer the "question". This query MUST be specific to the node's question, NOT the user's original question.
5.  "categories": A list of short, mutually exclusive categories for the answer, chosen from the AXES.
6.  "dependencies": A list of 'id's that this sub-question depends on.

The final node *must* have the "id" "final_conclusion" and the "axis" "integration".

---
AVAILABLE AXES AND CATEGORIES:
{axes_description}
---

USER QUESTION: "is suicide morally permissible?"

EXAMPLE GRAPH:
[
  {{
    "id": "q1_metaphysical_status",
    "axis": "metaphysical_status",
    "question": "What is the metaphysical status of suicide in relation to the Will-to-Live?",
    "search_query": "Schopenhauer suicide as affirmation or denial of the Will",
    "categories": ["affirms_will", "denies_will", "will_less", "neutral"],
    "dependencies": []
  }},
  {{
    "id": "q2_liberation_status",
    "axis": "liberation_status",
    "question": "Does suicide provide true liberation or escape from the Will and suffering?",
    "search_query": "Schopenhauer suicide escape from suffering or liberation",
    "categories": ["liberating", "futile", "deluded", "obscuring"],
    "dependencies": ["q1_metaphysical_status"]
  }},
  {{
    "id": "q3_ethical_status",
    "axis": "ethical_status",
    "question": "What is Schopenhauer's final moral judgment on suicide?",
    "search_query": "Schopenhauer moral view on suicide",
    "categories": ["morally_wrong", "morally_neutral", "outside_morality", "ascetic_good"],
    "dependencies": []
  }},
  {{
    "id": "final_conclusion",
    "axis": "integration",
    "question": "Integrate the metaphysical, liberation, and ethical findings into a final answer.",
    "search_query": "Schopenhauer summary on suicide",
    "categories": [],
    "dependencies": ["q1_metaphysical_status", "q2_liberation_status", "q3_ethical_status"]
  }}
]
---

USER QUESTION: "{question}"

GRAPH:
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
            keep_alive=-1,
        )
        raw_content = response["message"]["content"]
        
        # Extract JSON from the LLM's response
        graph = _extract_json_from_text(raw_content)
        
        if isinstance(graph, list) and all(isinstance(n, dict) for n in graph) and any(n.get('id') == 'final_conclusion' for n in graph):
            # Graph seems valid, return it
            return graph
        else:
            print(f"Warning: LLM did not return a valid graph. Falling back to simple graph.")
            # Fallback for simple questions
            return [
                { 
                  "id": "final_conclusion", 
                  "axis": "General",
                  "question": question, 
                  "search_query": question, 
                  "categories": ["Direct Answer"],
                  "dependencies": [] 
                }
            ]
            
    except Exception as e:
        print(f"Error during graph generation: {e}")
        return [
            { 
              "id": "final_conclusion", 
              "axis": "General",
              "question": question, 
              "search_query": question,
              "categories": ["Direct Answer"], 
              "dependencies": [] 
            }
        ]

if __name__ == "__main__":
    test_question = "is suicide morally permissible?"
    graph = generate_reasoning_graph(test_question)
    print(f"--- Reasoning Graph for: '{test_question}' ---")
    print(json.dumps(graph, indent=2))