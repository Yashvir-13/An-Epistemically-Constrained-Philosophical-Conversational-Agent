import json
import ollama
from pydantic import BaseModel, ValidationError
from typing import List, Literal

AXES = [
    "metaphysical_status",
    "ethical_status",
    "psychological_cause",
    "liberation_status",
    "aesthetic_effect",
    "religious_alignment"
]

SYSTEM_PROMPT = """
You are an AXIS ROUTER for a Schopenhauerian reasoning system.

Your task:
Rank ALL axes by relevance to the QUESTION.
Do NOT collapse everything into metaphysics.
Think in terms of: what must be explained FIRST.

Guidelines:
- Art, music, beauty, sublime, architecture, poetry → aesthetic_effect
- Desire, boredom, suffering, motivation → psychological_cause
- Right/wrong, justice, suicide → ethical_status
- Salvation, asceticism, denial of will → liberation_status
- Ontological essence of Will/Representation → metaphysical_status
- Religion, myth, Buddhism → religious_alignment

NEGATIVE CONSTRAINTS (STRICT):
1. SUICIDE is an ETHICAL problem (affirming the will), NOT metaphysical.
2. BOREDOM/SUFFERING are PSYCHOLOGICAL states first.
3. JUSTICE/WRONG-DOING are ETHICAL.
4. ONLY route to 'metaphysical_status' if the question asks about the Thing-in-Itself, The Will as a whole, or the nature of Reality.

IMPORTANT JSON SAFETY:
- In the "justification" field, DO NOT use double quotes (") or single quotes ('). Use plain text only.

Return a JSON object with a single key "axis_ranking" containing a list of objects.
Example:
{
  "axis_ranking": [
    {"axis": "metaphysical_status", "score": 0.8, "justification": "..."},
    {"axis": "psychological_cause", "score": 0.2, "justification": "..."}
  ]
}
Scores must sum to 1.0.
"""

class AxisRank(BaseModel):
    axis: Literal[
        "metaphysical_status",
        "ethical_status",
        "psychological_cause",
        "liberation_status",
        "aesthetic_effect",
        "religious_alignment"
    ]
    score: float
    justification: str

class AxisRouterOutput(BaseModel):
    axis_ranking: List[AxisRank]

class LexicalNecessity:
    """
    A Deterministic Rule Layer that 'forces' specific axes based on keywords.
    
    This exists to prevent the LLM from hallucinating "Psychology" when the user asks about
    conceptually distinct topics like "Suicide" (Ethics) or "Music" (Aesthetics).
    
    Necessity = If the word exists, the axis MUST be included.
    """
    @staticmethod
    def check_necessity(question: str) -> List[AxisRank]:
        q_lower = question.lower()
        forced_axes = []

        # 1. AESTHETICS (Book 3: Platonics Ideas, Genius, Art)
        aesthetic_terms = [
            "art", "music", "beauty", "beauti", "sublime", "tragedy", "genius", "aesthetic", 
            "architecture", "poetry", "painting", "sculpture", "melody", "harmony", "rhythm",
            "platonic idea", "eternal form"
        ]
        if any(w in q_lower for w in aesthetic_terms):
            forced_axes.append(AxisRank(
                axis="aesthetic_effect",
                score=0.95,
                justification="Necessity: Question concerns aesthetic contemplation, art, or the Ideas."
            ))

        # 2. ETHICS (Book 4: Affirmation/Denial, Justice, Egoism)
        ethical_terms = [
            "suicide", "justice", "wrong", "right", "moral", "ascetic", "compassion", 
            "egoism", "malice", "schadenfreude", "redemption", "salvation", "sainthood",
            "affirmation of the will", "denial of the will", "affirm the will", "deny the will",
            "quieter", "quietism"
        ]
        if any(w in q_lower for w in ethical_terms):
            # If specifically normative ("is it wrong"), boost higher
            score = 0.9 if "wrong" in q_lower or "right" in q_lower else 0.85
            forced_axes.append(AxisRank(
                axis="ethical_status",
                score=score,
                justification="Necessity: Question concerns normative ethics or negation of will."
            ))

        # 3. METAPHYSICS (Book 1 & 2: Representation, Will, Ontology)
        metaphysical_terms = [
            "thing-in-itself", "noumena", "will as a whole", "representation", "unity", "reality", 
            "metaphysic", "ontology", "epistemology", "causality", "space", "time", 
            "principium individuationis", "subject", "object", "matter", "force", "transcendental"
        ]
        if any(w in q_lower for w in metaphysical_terms):
            forced_axes.append(AxisRank(
                axis="metaphysical_status",
                score=0.8,
                justification="Necessity: Question concerns fundamental ontology or epistemology."
            ))

        return forced_axes

class AxisRouter:
    """
    Routes user questions to the appropriate philosophical domain (Axis).
    
    The routing pipeline is:
    1. **Necessity Check**: Look for mandatory keywords (e.g. "Music" -> Aesthetics).
    2. **LLM Ranking**: Ask Llama 3 to rank axes by relevance.
    3. **Merger**: Fuse the Necessity rules with the LLM output.
    4. **Fallback**: If all else fails, default to Psychology (the phenomenology of experience).
    """
    def __init__(self, model="llama3"):
        self.model = model

    def rank_axes(self, question: str) -> dict:
        """
        Determines which axes (Metaphysics, Ethics, etc.) are active for a question.
        
        Args:
            question (str): User input.
            
        Returns:
            dict: {"axis_ranking": [{"axis": "...", "score": ...}, ...]}
        """
        # 1. Check Necessity Rules FIRST
        necessity_ranks = LexicalNecessity.check_necessity(question)
        
        # If we have a clear primary necessity (only 1 or a strong winner), we can potentially return early 
        # or mix it with LLM. For safety/speed, if necessity matches, we rely on it heavily.
        
        prompt = f"""
QUESTION:
"{question}"

Return ranked axes with probabilities.
"""
        llm_result = None

        for _ in range(3):
            try:
                res = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    format="json",
                    options={"temperature": 0.0}
                )
                parsed = json.loads(res["message"]["content"])
                validated = AxisRouterOutput(**parsed)
                llm_result = validated.axis_ranking
                break
            except (ValidationError, json.JSONDecodeError):
                continue
        
        # Merge or Fallback
        final_ranking = []
        
        if llm_result:
            # If LLM worked, we still inject necessity to boost/ensure presence
            # Create a dict for easy lookup
            llm_map = {item.axis: item for item in llm_result}
            
            for nec in necessity_ranks:
                # Overwrite LLM score if necessity is found
                llm_map[nec.axis] = nec
            
            final_ranking = list(llm_map.values())
        else:
            # LLM Failed. Use Necessity or Psychological Default
            if necessity_ranks:
                final_ranking = necessity_ranks
                # Fill remaining probability with psychology if space exists
                total_nec = sum(n.score for n in necessity_ranks)
                if total_nec < 1.0:
                    rem = 1.0 - total_nec
                    final_ranking.append(AxisRank(
                        axis="psychological_cause",
                        score=rem,
                        justification="Fallback: Filling remaining probability."
                    ))
            else:
                # Pure Fallback (No Necessity, No LLM) -> Psychology
                return {
                    "axis_ranking": [
                        {
                            "axis": "psychological_cause",
                            "score": 0.8,
                            "justification": "Fallback: Default to phenomena/psychology."
                        },
                        {
                            "axis": "metaphysical_status",
                            "score": 0.2,
                            "justification": "Fallback: Secondary grounding."
                        }
                    ]
                }

        # Normalize and Sort
        # Ensure we have a list
        total = sum(a.score for a in final_ranking)
        if total == 0: total = 1.0 # prevent divy by zero
        
        for a in final_ranking:
            a.score = round(a.score / total, 4)
            
        # Sort descending
        final_ranking.sort(key=lambda x: x.score, reverse=True)

        return {"axis_ranking": [r.model_dump() for r in final_ranking]}
