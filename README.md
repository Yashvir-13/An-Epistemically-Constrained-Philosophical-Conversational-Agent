# Fathom: An Epistemically Constrained Philosophical Conversational Agent

Fathom is a sophisticated cognitive architecture simulation designed to model the *structure* of a philosopher's thought, specifically Arthur Schopenhauer. Unlike standard LLM chatbots, Fathom constrains its outputs using a probabilistic soft logic network and an epistemic profiler, ensuring that the agent's "beliefs" remain consistent with the philosopher's core axioms (Metaphysics, Ethics, Aesthetics).

## Core Architecture

![Fathom Architecture](architecture_diagram.png)

The architecture comprises three phases:

1.  **Offline Data Preparation**: A pipeline that cleans raw philosophical texts and enriches them with LLM-derived belief states to create a structured knowledge base.
2.  **Knowledge Mining**: A phase that computes global priors via Bayesian frequency counting and learns cross-axis correlations using Pointwise Mutual Information (PMI) to model how beliefs in one domain (e.g., Metaphysics) influence another (e.g., Ethics).
3.  **Runtime Inference Engine**: An iterative loop that integrates Retrieval-Augmented Generation (RAG) with a custom **Soft Logic Network** for probabilistic reasoning. This process is gated by an **Epistemic Profiler** that enforces tier-based linguistic permissions (e.g., preventing "weak" beliefs from being stated as "facts").

## Key Features

-   **Axis Routing**: The system decomposes user queries into specific philosophical domains (Metaphysics, Ethics, Aesthetics, etc.) to target retrieval and reasoning.
-   **Probabilistic Soft Logic**: Uses a custom `SoftLogicNetwork` (`probabilistic_reasoner.py`) to update beliefs based on evidence and cross-axis correlations, rather than simple vector similarity.
-   **Epistemic Profiling**: Calculates the entropy of belief distributions to determine the agent's confidence. High entropy (confusion) restricts the agent from making bold metaphysical claims.
-   **Dynamic Persona Synthesis**: The final response is synthesized by an LLM (Llama 3) that adopts a "Grumpy Schopenhauer" persona, conditioned on the rigorous logic and constraints derived from the reasoning engine.

## Installation & Setup

### Prerequisites
-   **Python 3.10+**
-   **Ollama**: With the `llama3` model pulled (`ollama pull llama3`).
-   **NVIDIA GPU**: Recommended for local embeddings (e.g., `e5-large-v2`).

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/fathom.git
    cd fathom
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure Ollama is running:
    ```bash
    ollama serve
    ```

## Usage

### Interactive Mode
Run the main agent to converse with the simulated philosopher:
```bash
python agent.py
```
Type your philosophical questions (e.g., "Is suicide morally permissible?") and observe the agent's reasoning process, epistemic status, and final synthesized answer.

### Batch Benchmark
Run the agent against a dataset of questions to evaluate its performance and reasoning consistency:
```bash
python run_agent_batch.py
```
This will output a `benchmark_results_v2.json` file containing the full reasoning traces and answers.

## Project Structure

-   `agent.py`: The main entry point and orchestration script.
-   `evidence_gatherer.py`: Handles RAG (Retrieval Augmented Generation) and vector database interactions.
-   `probabilistic_reasoner.py`: Implements the Soft Logic Network and Epistemic Profiling logic.
-   `reasoning_graph.py`: Manages the "Tree of Thought" reasoning steps (legacy/refactored).
-   `heurarchial_splitting.py`: Defines the philosophical axes and categories.
-   `requirements.txt`: Python package dependencies.

## License
MIT
