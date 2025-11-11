Agentic Weather + RAG Pipeline
This project is a Streamlit-based RAG application that allows users to upload PDF documents, process and semantically chunk them into a vector database, and query both the documents and live weather using Retrieval Augmented Generation. It supports immediate PDF ingestion, semantic chunking with overlap, and fast question answering.

Features
Upload PDFs: Drag-and-drop or browse to upload documents.
Instant Processing: Files are chunked and stored in the vector DB as soon as they are uploaded.
Semantic Chunking: Uses chunk overlap and semantic splitting for high-quality retrieval.
Hybrid Agent: Handles both weather queries (via API) and document questions in the same UI.
Document Search: All uploaded documents immediately available for RAG search.

LangChain & Qdrant Integration: Modular, fast, and scalable open-source vector search.

Project Structure

Agentic-AI-Pipeline/
│
├── src/
│   ├── app.py                 # Main Streamlit app
│   ├── config.py              # API keys and environment settings
│   ├── documentloader.py      # PDF loader and chunker
│   ├── graph.py               # Agentic logic and workflow
│   ├── langsmithevaluator.py  # Optional evaluation logic
│   ├── llmprovider.py         # LLM and embedding config
│   ├── vectordb.py            # Vector DB manager (e.g. Qdrant/Chroma)
│   ├── weatherapi.py          # External weather API integration
│
├── requirements.txt
└── README.md

Installation & Setup
Clone the repository:

text
git clone https://github.com/speedbar/Agentic-AI-Pipeline.git

Install dependencies:

text
pip install -r requirements.txt
Set environment variables:

Copy .env.example to .env and fill in your details for APIs (Azure OpenAI, OpenWeatherMap, Vector DB, etc.)

Example:

text
AZURE_OPENAI_API_KEY=...
OPENWEATHERMAP_API_KEY=...
VECTORDB_PATH=.chromadb

cd src
Run the app:
streamlit run src/app.py


📝 Usage
Use the sidebar to upload one or more PDF files.
Documents are processed/chunked and added to the vector DB immediately.
Ask questions about your documents or the weather in the chat interface.
See vector DB statistics and adjust settings live in the sidebar.

🧩 Customization
Switch to other chunking strategies by editing DocumentProcessor in documentloader.py.
Change vector DB backend (Qdrant, Chroma, FAISS) via vectordb.py/config.py.
You can tune chunk size and overlap for your use-case.
