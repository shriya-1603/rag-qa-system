# RAG Document Q&A

[![Streamlit App](https://static.streamlit.io/badge-streamlit.svg)](https://rag-app-system.streamlit.app/)

A fully local or cloud-enabled Retrieval-Augmented Generation (RAG) system for asking questions about your documents. Upload PDFs, Word files, or plain text, index them into a FAISS vector store, and chat with them using either a local Llama 3 model (via Ollama) or a cloud-hosted Gemini 3.5 Flash model.

**Deployed Link:** [https://rag-app-system.streamlit.app/](https://rag-app-system.streamlit.app/)

---

## 🌟 Dual-Mode Execution

This system automatically detects its environment and adapts for seamless developer and user workflows:

*   **Local Mode (Default):** Runs fully locally on your machine. LLM generation and embeddings are powered by Ollama (`llama3` and `nomic-embed-text`). No API keys required, completely private.
*   **Cloud Mode:** Activated automatically when the `GEMINI_API_KEY` environment variable is set. It switches to Google's cloud-based models (`gemini-3.5-flash` and `gemini-embedding-001`) for ultra-low latency, wider context, and server-less execution.

---

## 📊 Technical Specifications & Metrics

Below are the benchmark configurations, performance metrics, and model characteristics:

### ⚙️ RAG Configuration Parameters
*   **Chunk Size:** `1000` characters (~150-250 words per chunk). Ideal for preserving context while keeping individual segments compact.
*   **Chunk Overlap:** `200` characters (~30-50 words). Prevents loss of information at the borders of text boundaries.
*   **Retrieval Depth (Top-K):** `4` chunks. Balances comprehensive context coverage with LLM prompt context window economy.

### 📈 Model & Embedding Comparisons

| Feature / Metric | Local Mode (Ollama) | Cloud Mode (Google Gemini) |
| :--- | :--- | :--- |
| **Generator LLM** | `llama3` (8B) | `gemini-3.5-flash` |
| **Embedding Model** | `nomic-embed-text` | `gemini-embedding-001` |
| **Embedding Dimension** | 768 dimensions | 3072 dimensions |
| **Index Search Latency** | < 2 ms (FAISS CPU) | < 2 ms (FAISS CPU) |
| **Avg. Retrieval + Gen Latency**| ~3.5 - 7.5 seconds (depends on hardware) | **~1.2 - 2.5 seconds** |
| **Context Window Size** | 8,192 tokens | **1,000,000+ tokens** |

### 🛠️ Ingestion Throughput (Approximate)
*   **Text/TXT Loading:** ~10,000 words/second.
*   **PDF/DOCX Processing:** ~10-15 pages/second (depends on CPU power for text extraction).
*   **Vector Store Building:** FAISS CPU indexing takes <10ms for up to 1,000 document chunks.

---

## 🛠️ Setup & Local Run

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed and running (for Local Mode)

Pull the required local models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Installation
```bash
# Clone the repository
git clone https://github.com/shriya-1603/rag-qa-system.git
cd rag-qa-system

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Cloud Deployment Configuration

When deploying to platforms like **Streamlit Community Cloud**:
1. Add your repository to the dashboard.
2. In the **Advanced settings... -> Secrets** section, add your API key:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
3. Save and deploy. The app will automatically connect to Gemini and transition into Cloud Mode.
