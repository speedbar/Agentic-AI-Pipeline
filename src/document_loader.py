from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_db import VectorDatabaseManager
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_db = VectorDatabaseManager()
    
    def load_pdf(self, pdf_path: str):
        # \"\"\"Load and process PDF\"\"\"
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
        
        # Split documents
        split_docs = self.splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Add to vector store
        self.vector_db.add_documents(split_docs)
        return split_docs
    
    def load_multiple_pdfs(self, pdf_paths: list):
        # \"\"\"Load multiple PDFs\"\"\"
        all_docs = []
        for pdf_path in pdf_paths:
            docs = self.load_pdf(pdf_path)
            all_docs.extend(docs)
        return all_docs