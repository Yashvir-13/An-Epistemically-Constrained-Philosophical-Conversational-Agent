
import json
import ollama
from axis_router import AxisRouter

def probe():
    """
    A Quick Diagnostics Tool for the Axis Router.
    
    It runs a set of standard questions covering all 4 books of 'The World as Will and Representation'
    through the `AxisRouter` logic and prints the routing decision (Necessity vs LLM).
    
    Useful for verifying routing logic changes without running the full agent.
    """
    router = AxisRouter()
    
    # Diverse set of questions covering the "Four Books" of World as Will and Representation
    questions = [
        # Book 1: Epistemology / Metaphysics
        "Is the world merely my representation?",
        "What is the role of time and space in our understanding?",
        
        # Book 2: Will / Metaphysics of Nature
        "Is gravity a manifestation of the Will?",
        "What is the thing-in-itself?",
        
        # Book 3: Aesthetics
        "Why does the genius suffer more than the common man?",
        "How does tragedy reveal the nature of the will?",
        
        # Book 4: Ethics
        "Is egoism the natural state of man?",
        "Does the ascetic achieve salvation through starvation?",
        
        # Tricky / Multi-Axis
        "Is the platinum idea of beauty real?"
    ]

    print("--- Probing Axis Router (Generalized) ---\n")
    for q in questions:
        print(f"Q: {q}")
        res = router.rank_axes(q)
        print(json.dumps(res, indent=2))
        print("-" * 40)

if __name__ == "__main__":
    probe()
