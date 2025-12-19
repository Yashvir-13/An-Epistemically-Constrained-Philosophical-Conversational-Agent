# Fathom

**Fathom** is a research-oriented philosophical conversational system designed to model *reasoning* rather than surface-level text generation. The project focuses on **Arthur Schopenhauer** as a first case study and combines symbolic philosophy, probabilistic inference, and large language models into a unified pipeline.

Unlike conventional chatbots, Fathom decomposes philosophical questions into structured sub-problems, gathers evidence from primary texts, performs uncertainty-aware inference across multiple philosophical axes, and only then synthesizes a persona-constrained response.

This repository represents an experimental prototype aimed at exploring how philosophical understanding can be **explicitly represented, evaluated, and reasoned over**, rather than implicitly mimicked.

---

## Core Ideas

Fathom is built around a few guiding principles:

* **Reasoning over Generation** – The system treats language models as tools for analysis and annotation, not as authoritative sources of truth.
* **Explicit Philosophical Structure** – Concepts, axes, and categories are defined explicitly rather than left implicit in prompts.
* **Uncertainty Awareness** – Philosophical conclusions are treated probabilistically, not as absolute outputs.
* **Hallucination Resistance** – The system is designed to detect false premises, underdetermined questions, and missing evidence.
* **Persona as Constraint** – Philosophical voice (e.g., Schopenhauer) is applied *after* reasoning, not used to replace it.

---

## System Architecture

Fathom operates in two phases: **offline knowledge construction** and **online reasoning & response generation**.
<img width="547" height="438" alt="image" src="https://github.com/user-attachments/assets/332128b1-bc96-48e2-b550-886d5b33e31a" />


### 1. Offline Knowledge Construction

**Document Cleaning & Structuring**

* Primary texts (e.g., Project Gutenberg editions of Schopenhauer) are cleaned of boilerplate and parsed into structured sections (books, chapters, §§).
* Output: `structured_documents.json`

**Ontology-Guided Enrichment**

* Each text chunk is analyzed using an LLM to score ~80 Schopenhauerian concepts (e.g., *will-to-live*, *denial of the will*, *compassion*) on a scale from **-1.0 to +1.0**.
* The result is a *belief-state vector* per document, representing how strongly each concept is expressed or rejected.
* Output: `enriched_documents.json`

**Vector Database Construction**

* Enriched documents are embedded using `intfloat/e5-large-v2` and stored in a persistent Chroma vector database.

---

### 2. Online Reasoning Pipeline

**Reasoning Graph Generation**

* A user question is decomposed into a directed graph of sub-questions.
* Each node is assigned:

  * a philosophical axis (e.g., metaphysical, ethical, psychological)
  * mutually exclusive categorical answers
  * dependencies on other nodes
  <img width="346" height="395" alt="image" src="https://github.com/user-attachments/assets/a3b2741d-5f63-4b2e-9fc4-940427d9cb05" />


**Axis-Aware Evidence Gathering**

* Relevant text passages are retrieved using axis-specific query expansion.
* Retrieved evidence is re-ranked using:

  * axis relevance
  * subject alignment
  * anti-conflation penalties
  * cross-axis relevance penalties

**Probabilistic Reasoning**

* Prior distributions over philosophical categories are learned from the corpus itself.
* Evidence from each node is combined with priors using Bayesian-style log-space updates.
* Cross-axis consistency constraints (e.g., metaphysics ↔ ethics ↔ liberation) are applied to enforce philosophical coherence.

**Persona-Constrained Synthesis**

* Only after inference is complete does the system generate a response in the voice of a specific philosopher.
* Hard constraints prevent biographical errors, false attributions, or compliance with false premises.

---

## Evaluation Framework

Fathom includes a custom benchmark, **SchopenhauerBench**, designed to test philosophical *faithfulness* rather than fluency.

**Dataset**

* ~30 expert-style questions generated from authoritative secondary sources (e.g., IEP).
* Each question includes:

  * required factual points
  * adversarial (false-premise) labels where applicable

**Metrics**

* Answers are graded by an independent LLM acting as an academic judge.
* Scoring focuses on:

  * factual alignment with required points
  * correct rejection of false premises
  * absence of hallucinated moral/religious claims

**Current Results**

<img width="704" height="494" alt="image" src="https://github.com/user-attachments/assets/2d68271c-e600-46d9-be20-df0d4b84f600" />

* Overall Faithfulness: ~4.7 / 5.0
* Strong adversarial resistance, with remaining failures primarily in underrepresented formal topics (e.g., Principle of Sufficient Reason).

---

## Limitations (Known Issues)

* **Philosopher-Specific Design** – Ontology and consistency rules are currently tailored to Schopenhauer.
* **Evaluation Rigor** – Current scoring relies on LLM-based judging; expert-curated datasets are a future goal.
* **Formal Domains** – Structural metaphysics (e.g., PSR, space/time formalism) require additional axes.
* **Manual Consistency Rules** – Cross-axis constraints are hand-designed, not learned.

These limitations are explicit and intentional areas of future work.

---

## Intended Audience

This project is intended for:

* Philosophy researchers interested in computational modeling
* AI researchers working on neuro-symbolic or explainable systems
* Digital humanities scholars
* Educators interested in structured philosophical evaluation

It is **not** intended as a production chatbot.

---

## Project Status

Fathom is an active research prototype.

Future directions include:

* Philosopher-agnostic ontology support
* Expert-curated evaluation datasets
* Formal abstention / underdetermination detection
* Comparative reasoning across philosophical traditions

---

## Author

**Yashvir**
Undergraduate researcher exploring philosophical reasoning, probabilistic inference, and AI alignment with human intellectual traditions.

---

*This repository is shared for academic discussion, critique, and collaboration.*
