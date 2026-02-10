# doctrine_anchors.py

# doctrine_anchors.py

"""
Defines the SYSTEM-LEVEL CONSTANTS for 'Non-Negotiable Doctrines'.

These are the core axioms of Schopenhauer's system that the agent MUST NEVER contradict.
Used by the `apply_doctrinal_corrections` function in `agent.py` and by the `score_benchmark.py`
metric checks.
"""

NON_NEGOTIABLE_DOCTRINES = {
    "will_is_thing_in_itself": {
        "axis": "metaphysical_status",
        "keywords": ["will", "thing-in-itself", "noumena"],
        "required_category": "neutral", # In Fathom's 4-cat axis, 'neutral' is the base for the essence itself
        "entropy_threshold": 0.8
    },
    "suicide_not_wrong": {
        "axis": "ethical_status",
        "keywords": ["suicide", "wrong", "sin"],
        "required_category": "outside_morality",
        "entropy_threshold": 0.8
    },
    "ascetic_negation": {
        "axis": "metaphysical_status",
        "keywords": ["ascetic", "starvation", "negation", "denial"],
        "required_category": "denies_will",
        "entropy_threshold": 0.8
    },
    "aesthetic_suspension": {
        "axis": "aesthetic_effect",
        "keywords": ["aesthetic", "contemplation", "beauty"],
        "required_category": "will_less_contemplation",
        "entropy_threshold": 0.8
    }
}
