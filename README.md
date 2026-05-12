⏺ # RAG Document Q&A                                  
                                                     
  A fully local Retrieval-Augmented Generation (RAG)  
  system for asking questions about your documents.
  Upload PDFs, Word files, or plain text, index them  
  into a FAISS vector store, and chat with them via a
  Llama 3 model — all running on your machine, no API 
  keys required.                                      
                                                      
  Built with LangChain, FAISS, Ollama, and Streamlit.

  ---

  ## Features

  - **Fully local** — LLM and embeddings run via      
  Ollama; nothing leaves your machine
  - **Multi-format ingestion** — supports PDF, DOCX,  
  and TXT         
  - **Persistent vector index** — FAISS store is saved
   to disk and reused across sessions
  - **Source citations** — answers link back to the
  source document and page number                     
  - **Dark / light mode** — toggle in the sidebar
                                                      
  ---             
                                                    
  ## Prerequisites

  - Python 3.9+
  - [Ollama](https://ollama.com) installed and running

  Pull the required models before first run:

  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text                        
   
  ---                                                 
  Setup           
                                                    
  # Clone the repo
  git clone
  https://github.com/shriya-1603/rag-qa-system.git
  cd rag-qa-system

  # Create and activate a virtual environment         
  python -m venv .venv
  source .venv/bin/activate   # Windows:              
  .venv\Scripts\activate
                                                    
  # Install dependencies
  pip install -r requirements.txt

  ---
  Running the app

  streamlit run streamlit_app.py
                                                      
  Open http://localhost:8501 in your browser.
                                                      
  1. Upload one or more documents in the sidebar (PDF,
   DOCX, or TXT).                                   
  2. Click Index Documents to chunk and embed them
  into the vector store.
  3. Type a question in the chat input and press
  Enter.                                              
   
  ---                                                 
  Configuration   
                                                    
  All settings can be overridden with environment
  variables (or a .env file):

  Variable: LLM_MODEL                              
  Default: llama3                                  
  Description: Ollama model used for generation    
  ────────────────────────────────────────         
  Variable: EMBEDDING_MODEL                        
  Default: nomic-embed-text                           
  Description: Ollama model used for embeddings    
  ────────────────────────────────────────            
  Variable: OLLAMA_BASE_URL                        
  Default: http://localhost:11434                   
  Description: Ollama server URL                   
  ────────────────────────────────────────         
  Variable: CHUNK_SIZE                             
  Default: 1000                                    
  Description: Characters per document chunk
  ────────────────────────────────────────            
  Variable: CHUNK_OVERLAP
  Default: 200                                        
  Description: Overlap between consecutive chunks
  ────────────────────────────────────────          
  Variable: TOP_K
  Default: 4
  Description: Number of chunks retrieved per query

  ---
  Project structure
                                                      
  rag-qa-system/
  ├── app/                                            
  │   ├── ingestion.py     # Document loading and
  chunking                                          
  │   ├── embeddings.py    # Embedding helpers
  │   ├── vectorstore.py   # FAISS index management
  │   └── qa_chain.py      # RetrievalQA chain        
  construction
  ├── data/                                           
  │   ├── raw/             # Uploaded source files
  │   └── vectorstore/     # Persisted FAISS index  
  ├── config.py            # Centralised config / env
  vars                                                
  ├── streamlit_app.py     # Streamlit UI
  └── requirements.txt                                
                  
  ---                                               
  Tech stack

  ┌───────────────┬───────────────────────────┐
  │   Component   │          Library          │
  ├───────────────┼───────────────────────────┤
  │ LLM           │ Ollama (llama3)           │
  ├───────────────┼───────────────────────────┤
  │ Embeddings    │ Ollama (nomic-embed-text) │
  ├───────────────┼───────────────────────────┤       
  │ Vector store  │ FAISS                     │
  ├───────────────┼───────────────────────────┤       
  │ Orchestration │ LangChain                 │
  ├───────────────┼───────────────────────────┤     
  │ UI            │ Streamlit                 │
  └───────────────┴───────────────────────────┘
