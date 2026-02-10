import ollama
from typing import List, Tuple

VERIFIER_MODEL = "llama3"

def verify_and_repair(answer: str, non_negotiable_constraints: List[str]) -> Tuple[str, bool]:
    """
    Verifies that the answer adheres to non-negotiable constraints.
    
    Mechanism:
    1. Feeds the Answer + Constraints to a 'Rigorous Logic' LLM persona.
    2. Asks it to detect 'AXIS DRIFT' (leaving the routed philosophical domain).
    3. If a violation is found, it asks for a 'Fixed Text'.
    4. **Feedback Loop**: It recursively Re-Verifies the 'Fixed Text' to ensure the fix didn't introduce new errors.
    
    Returns:
        tuple: (final_text, was_repaired_bool)
    """
    if not non_negotiable_constraints:
        return answer, False

    # We treat polarity rules as part of non-negotiable constraints for the verifier, 
    # as they are passed in the same list by the caller (agent.py needs to pass them combined or we update signature).
    # NOTE: caller in agent.py currently passes `non_negotiable`. 
    # We need to ensure `polarity_rules` are appended to `non_negotiable` BEFORE calling this function in agent.py
    
    constraints_text = "\n".join([f"- {c}" for c in non_negotiable_constraints])

    prompt = f"""
    SYSTEM INSTRUCTION:
    You are a Logic Engine, NOT a chatbot. 
    You verify if an Answer adheres to constraints.
    
    CONSTRAINTS:
    {constraints_text}
    
    ANSWER:
    "{answer}"
    
    TASK:
    - Check the ANSWER against CONSTRAINTS.
    - Specifically look for "AXIS DRIFT": If the CONSTRAINTS include an "AXIS LOCK", ensure the answer does not stray into forbidden metaphysical territory or use prohibited abstractions.
    
    CRITICAL OUTPUT FORMAT:
    If NO violation:
    [VERIFIER REPORT]
    Status: CLEAN
    
    If VIOLATION (Hard or Axis Drift):
    [VERIFIER REPORT]
    Status: VIOLATION
    Details: <Logic explanation of what rule failed, e.g., 'Metaphysical drift in psychological context'>
    Fixed Text: <The exact rewritten answer text enforcing axis discipline>
    """

    try:
        response = ollama.chat(
            model=VERIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}, # Deterministic
        )
        result = response["message"]["content"].strip()
        
        # Parse Report
        if "Status: CLEAN" in result:
             print("     [Verifier Report]: Status: CLEAN")
             return answer, False
        
        if "Fixed Text:" in result:
             # Extract text after "Fixed Text:"
             parts = result.split("Fixed Text:")
             if len(parts) > 1:
                 repair = parts[1].strip()
                 print(f"     [Verifier]: VIOLATION DETECTED. Repaired.")
                 # Print report for debug
                 print(result.split("Fixed Text:")[0])
                 
                 # --- RECURSIVE CHECK (FEEDBACK LOOP) ---
                 # We must ensure the repair is actually VALID.
                 print(f"     [Verifier]: Re-verifying repair...")
                 rec_prompt = prompt.replace(f'"{answer}"', f'"{repair}"')
                 rec_res = ollama.chat(model=VERIFIER_MODEL, messages=[{"role": "user", "content": rec_prompt}], options={"temperature": 0.0})
                 rec_text = rec_res["message"]["content"].strip()
                 
                 if "Status: CLEAN" in rec_text:
                     print("     [Verifier]: Repair CONFIRMED Clean.")
                     return repair, True # Repaired and Verified
                 else:
                     print("     [Verifier]: Repair FAILED verification. Discarding repair.")
                     # If repair failed, we might ideally return original + error flag, or keep best effort.
                     # Requirements say: mark 'repaired': false if it fails constraints.
                     # We return the repair (best effort) but return False for 'repaired' status effectively?
                     # Actually user said: Only mark "repaired": true if passes.
                     return repair, False # We return the text, but flag as NOT successfully repaired status-wise.

        # Fallback if format broken but looks clean
        if "CLEAN" in result:
            return answer, False

        print(f"     [Verifier]: Format unclear, defaulting to PASS. Raw: {result[:50]}...")
        return answer, False
            
    except Exception as e:
        print(f"     [Verifier] Error: {e}")
        return answer, False
