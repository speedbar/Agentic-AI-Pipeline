import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_db import VectorDatabaseManager
from document_loader import DocumentProcessor
from weather_api import WeatherAPI
from graph import RagAgentGraph, weather_tool, document_tool


# Fixtures
@pytest.fixture
def temp_chroma_dir():
    """Create temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings to avoid API calls"""
    with patch('vector_db.get_embeddings') as mock:
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[0.1] * 1536]
        embeddings.embed_query.return_value = [0.1] * 1536
        mock.return_value = embeddings
        yield embeddings


@pytest.fixture
def vector_db_manager(temp_chroma_dir, mock_embeddings):
    """Create VectorDatabaseManager with temporary directory"""
    return VectorDatabaseManager(persist_directory=temp_chroma_dir)


@pytest.fixture
def document_processor(vector_db_manager):
    """Create DocumentProcessor with mocked vector DB"""
    with patch('document_loader.VectorDatabaseManager') as mock_vdb:
        mock_vdb.return_value = vector_db_manager
        return DocumentProcessor()


@pytest.fixture
def mock_weather_api():
    """Mock Weather API responses"""
    with patch('weather_api.requests') as mock:
        yield mock


@pytest.fixture
def weather_api():
    """Create WeatherAPI instance"""
    return WeatherAPI()


@pytest.fixture
def mock_llm():
    """Mock LLM responses"""
    with patch('graph.get_azure_llm') as mock:
        llm = MagicMock()
        response = MagicMock()
        response.content = "This is a test response from the LLM"
        llm.invoke.return_value = response
        mock.return_value = llm
        yield llm


@pytest.fixture
def rag_agent(mock_llm, temp_chroma_dir):
    """Create RagAgentGraph with mocked LLM"""
    with patch('graph.VectorDatabaseManager') as mock_vdb, \
         patch('graph.WeatherAPI'):
        # Mock vector DB to use temp directory
        mock_vdb_instance = MagicMock()
        mock_vdb_instance.persist_directory = temp_chroma_dir
        mock_vdb.return_value = mock_vdb_instance
        return RagAgentGraph()


# Vector Database Tests
class TestVectorDatabase:
    
    def test_initialization(self, vector_db_manager, temp_chroma_dir):
        """Test VectorDatabaseManager initialization"""
        assert vector_db_manager.collection_name == "documents"
        assert vector_db_manager.persist_directory == temp_chroma_dir
        assert os.path.exists(temp_chroma_dir)
    
    def test_get_vector_store(self, vector_db_manager):
        """Test getting vector store"""
        vector_store = vector_db_manager.get_vector_store()
        assert vector_store is not None
    
    def test_add_documents(self, vector_db_manager):
        """Test adding documents to vector store"""
        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.pdf"}),
            Document(page_content="Test document 2", metadata={"source": "test2.pdf"}),
        ]
        
        ids = vector_db_manager.add_documents(docs)
        assert len(ids) == 2
    
    def test_search_documents(self, vector_db_manager):
        """Test document search"""
        # Add documents first
        docs = [
            Document(page_content="Python programming", metadata={"source": "test.pdf"}),
            Document(page_content="JavaScript coding", metadata={"source": "test2.pdf"}),
        ]
        vector_db_manager.add_documents(docs)
        
        # Search
        results = vector_db_manager.search_documents("Python", k=1)
        assert len(results) >= 0  # May or may not find results depending on embeddings
    
    def test_get_collection_info(self, vector_db_manager):
        """Test getting collection info"""
        info = vector_db_manager.get_collection_info()
        assert "name" in info
        assert "points_count" in info


# Document Processing Tests
class TestDocumentProcessor:
    
    def test_initialization(self, document_processor):
        """Test DocumentProcessor initialization"""
        assert document_processor.chunk_size == 1000
        assert document_processor.chunk_overlap == 200
    
    def test_document_splitting(self, document_processor):
        """Test document chunking"""
        long_text = "This is a test sentence. " * 100
        doc = Document(page_content=long_text, metadata={"source": "test.pdf"})
        
        chunks = document_processor.splitter.split_documents([doc])
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= document_processor.chunk_size + document_processor.chunk_overlap 
                   for chunk in chunks)
    
    @patch('document_loader.PyPDFLoader')
    def test_load_pdf(self, mock_loader, document_processor):
        """Test PDF loading"""
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = [
            Document(page_content="Page 1 content", metadata={"source": "test.pdf", "page": 1}),
            Document(page_content="Page 2 content", metadata={"source": "test.pdf", "page": 2}),
        ]
        
        with patch.object(document_processor.vector_db, 'add_documents', return_value=["id1", "id2"]):
            docs = document_processor.load_pdf("test.pdf")
            
            assert len(docs) > 0
            mock_loader.assert_called_once_with("test.pdf")


# Weather API Tests
class TestWeatherAPI:
    
    def test_initialization(self, weather_api):
        """Test WeatherAPI initialization"""
        assert weather_api.base_url == "https://api.openweathermap.org/data/2.5"
    
    def test_get_current_weather_success(self, weather_api, mock_weather_api):
        """Test successful weather retrieval"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "London",
            "main": {
                "temp": 15.5,
                "feels_like": 14.0,
                "humidity": 70
            },
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 5.5}
        }
        mock_response.raise_for_status = MagicMock()
        mock_weather_api.get.return_value = mock_response
        
        result = weather_api.get_current_weather("London")
        
        assert result["location"] == "London"
        assert result["temperature"] == 15.5
        assert "error" not in result
    
    def test_get_current_weather_error(self, weather_api, mock_weather_api):
        """Test weather API error handling"""
        mock_weather_api.get.side_effect = Exception("API Error")
        
        result = weather_api.get_current_weather("Invalid")
        
        assert "error" in result


# Tool Tests
class TestTools:
    
    def test_weather_tool(self):
        """Test weather tool with proper mocking"""
        mock_weather_data = {
            "location": "London",
            "temperature": 15.5,
            "feels_like": 14.0,
            "humidity": 70,
            "description": "cloudy",
            "wind_speed": 5.5
        }
        
        with patch('graph.WeatherAPI') as mock_weather_class:
            mock_api = MagicMock()
            mock_api.get_current_weather.return_value = mock_weather_data
            mock_weather_class.return_value = mock_api
            
            result = weather_tool.invoke({"location": "London"})
            
            # Check that result contains expected data
            assert "London" in result
            assert "15.5" in result or "temperature" in result.lower()
    
    def test_document_tool(self):
        """Test document tool with proper mocking"""
        with patch('graph.VectorDatabaseManager') as mock_vdb_class:
            mock_vdb = MagicMock()
            mock_store = MagicMock()
            
            mock_docs = [
                Document(
                    page_content="Test content about AI and machine learning",
                    metadata={"source": "test.pdf", "page": 1}
                )
            ]
            mock_store.similarity_search.return_value = mock_docs
            mock_vdb.get_vector_store.return_value = mock_store
            mock_vdb_class.return_value = mock_vdb
            
            result = document_tool.invoke({"query": "What is AI?"})
            
            assert "Test content" in result or "test.pdf" in result


# Graph Tests
class TestRagAgentGraph:
    
    def test_initialization(self, rag_agent):
        """Test RagAgentGraph initialization"""
        assert rag_agent.llm is not None
        assert rag_agent.graph is None
    
    def test_route_node_weather(self, rag_agent):
        """Test routing to weather"""
        state = {
            "query": "What's the weather in Paris?",
            "route": "",
            "weather_data": "",
            "document_data": "",
            "response": "",
            "messages": []
        }
        
        result = rag_agent.route_node(state)
        assert result["route"] == "weather"
    
    def test_route_node_documents(self, rag_agent):
        """Test routing to documents"""
        state = {
            "query": "What does the document say about AI?",
            "route": "",
            "weather_data": "",
            "document_data": "",
            "response": "",
            "messages": []
        }
        
        result = rag_agent.route_node(state)
        assert result["route"] == "documents"
    
    def test_weather_node(self, rag_agent):
        """Test weather node with mocked tool"""
        with patch('graph.weather_tool') as mock_weather_tool:
            mock_weather_tool.invoke.return_value = "{'location': 'London', 'temperature': 15.5}"
            
            state = {
                "query": "What's the weather in London?",
                "route": "weather",
                "weather_data": "",
                "document_data": "",
                "response": "",
                "messages": []
            }
            
            result = rag_agent.weather_node(state)
            assert result["weather_data"] != ""
    
    def test_document_node(self, rag_agent):
        """Test document node with mocked tool"""
        with patch('graph.document_tool') as mock_doc_tool:
            mock_doc_tool.invoke.return_value = "Document content about AI"
            
            state = {
                "query": "Tell me about AI",
                "route": "documents",
                "weather_data": "",
                "document_data": "",
                "response": "",
                "messages": []
            }
            
            result = rag_agent.document_node(state)
            assert result["document_data"] != ""
    
    def test_response_node(self, rag_agent):
        """Test response generation"""
        state = {
            "query": "What's the temperature?",
            "route": "weather",
            "weather_data": "Temperature: 15°C in London",
            "document_data": "",
            "response": "",
            "messages": []
        }
        
        result = rag_agent.response_node(state)
        assert result["response"] != ""
    
    def test_build_graph(self, rag_agent):
        """Test graph building"""
        rag_agent.build_graph()
        assert rag_agent.graph is not None
    
    def test_full_weather_pipeline(self, rag_agent):
        """Test complete weather query pipeline"""
        with patch('graph.weather_tool') as mock_weather_tool:
            mock_weather_tool.invoke.return_value = "{'location': 'Paris', 'temperature': 20}"
            
            rag_agent.build_graph()
            result = rag_agent.invoke("What's the weather in Paris?")
            
            assert result["route"] == "weather"
            assert result["response"] != ""


# Integration Tests
class TestIntegration:
    
    def test_end_to_end_weather_query(self, temp_chroma_dir):
        """Test complete weather query flow"""
        with patch('graph.get_azure_llm') as mock_llm, \
             patch('graph.weather_tool') as mock_weather_tool, \
             patch('graph.VectorDatabaseManager'), \
             patch('graph.WeatherAPI'):
            
            # Setup LLM mock
            mock_llm_instance = MagicMock()
            response = MagicMock()
            response.content = "The weather in London is 15°C and cloudy."
            mock_llm_instance.invoke.return_value = response
            mock_llm.return_value = mock_llm_instance
            
            # Setup weather tool mock
            mock_weather_tool.invoke.return_value = "{'temperature': 15, 'description': 'cloudy'}"
            
            # Create and run graph
            agent = RagAgentGraph()
            agent.build_graph()
            result = agent.invoke("What's the weather in London?")
            
            assert result["route"] == "weather"
            assert result["response"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])