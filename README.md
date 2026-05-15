# 🛡️ TruthLayer

Pioneering transparency and reliability: ensuring AI remains grounded and explainable.

**TruthLayer** is a local, privacy-first medical AI hallucination detection tool designed to ensure that AI-generated medical answers are trustworthy, consistent, and grounded in provided context. Built for the Gemma-4 Hackathon, TruthLayer uses local LLM inference via Ollama and a robust multi-pass verification system to assign a "Trust Score" to any medical query response.

## 🌟 Key Features

- **100% Local Privacy**: Runs entirely on your local machine using Ollama (supporting Gemma-4 and customized truthlayer-med models), ensuring no sensitive medical data is sent to external APIs.
- **Multi-Run Verification**: Mitigates hallucinations by querying the AI multiple times with varying temperatures. It checks the consistency across different generation paths to validate the answer.
- **Trust Scoring System**: Analyzes multiple generated responses and computes a comprehensive Trust Score (HIGH, MODERATE, or LOW) based on:
  - **Consistency**: Similarity across different runs.
  - **Confidence Language**: Detection of uncertainty or hedging keywords.
  - **Length Variance**: Consistency in the depth of the explanation.
- **Document Grounding (RAG)**: Upload medical PDFs and use Retrieval-Augmented Generation (powered by ChromaDB and Sentence-Transformers) to ground the AI's responses strictly in your trusted documents.
- **Reasoning Graph Visualization**: Demystifies the AI's logic by visualizing the step-by-step reasoning that led to the best answer.

## 🏗️ Project Structure

- `app.py`: The main Streamlit user interface.
- `core/`: The core logic of the application.
  - `trust_score.py`: Computes the final trust score based on various signals.
  - `consistency_checker.py`: Measures the semantic similarity between multiple model runs.
  - `confidence_detector.py`: Analyzes text for hedging and uncertainty.
  - `reasoning_graph.py`: Renders the step-by-step reasoning logic.
  - `ollama_client.py`: Handles communication with the local Ollama instance.
- `rag/`: Retrieval-Augmented Generation logic.
  - `retriever.py`: Manages document ingestion, chunking, and ChromaDB vector search.

## 🚀 Setup & Installation

### Prerequisites

1. **Python 3.8+** installed on your system.
2. **Ollama** installed and running locally ([Download Ollama](https://ollama.com/)).
3. Ensure you have pulled the required models in Ollama:
   ```bash
   ollama run gemma4:e4b
   ```

### Installation Steps

1. **Clone the repository** (if you haven't already) and navigate to the project directory:
   ```bash
   cd TruthLayer
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Mac/Linux/Bash:**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. Make sure your local **Ollama** instance is running in the background.
2. Ensure your virtual environment is activated.
3. Start the application:
   ```bash
   streamlit run app.py
   ```
4. Open the provided local URL in your browser (usually `http://localhost:8501`).
5. **(Optional)** Upload medical PDFs in the sidebar and enable "RAG" to ground the AI on your documents.
6. Enter a medical query, select your preferred model, adjust the number of verification runs, and hit Enter!

## 🛠️ Technology Stack

- **UI Framework:** Streamlit
- **Local LLM Runner:** Ollama (Gemma-4)
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence-Transformers
- **PDF Processing:** PyPDF
