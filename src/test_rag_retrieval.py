import pytest
from vector_db import VectorDatabaseManager
from document_loader import DocumentProcessor
from langchain_core.documents import Document

@pytest.fixture
def vector_db_manager():
    return VectorDatabaseManager()

@pytest.fixture
def document_processor():
    return DocumentProcessor()

def test_vector_store_initialization(vector_db_manager):
    """Test Qdrant collection initialization"""
    vector_db_manager.initialize_collection()
    assert vector_db_manager.collection_name is not None

def test_add_documents(vector_db_manager):
    """Test adding documents to vector store"""
    vector_db_manager.initialize_collection()
    
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test1"}),
        Document(page_content="Test document 2", metadata={"source": "test2"}),
    ]
    
    ids = vector_db_manager.add_documents(docs)
    assert len(ids) == 2

def test_document_splitting(document_processor):
    """Test document chunking"""
    long_text = "This is a test. " * 100
    doc = Document(page_content=long_text)
    
    chunks = document_processor.splitter.split_documents([doc])
    assert len(chunks) > 1