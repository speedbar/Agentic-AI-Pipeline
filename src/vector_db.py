from langchain_chroma import Chroma
from llm_provider import get_embeddings
import os
import logging

logger = logging.getLogger(__name__)


class VectorDatabaseManager:
    """Manages ChromaDB vector database operations - No Docker needed!"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB
        
        Args:
            persist_directory: Local directory to store the database (default: ./chroma_db)
        """
        self.persist_directory = persist_directory
        self.collection_name = "documents"
        self.embeddings = get_embeddings()
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        logger.info(f"ChromaDB initialized at: {persist_directory}")
    
    def get_vector_store(self):
        """Get Chroma vector store for LangChain
        
        Returns:
            Chroma vector store instance
        """
        try:
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            
            logger.info(f"Vector store ready with collection: {self.collection_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error getting vector store: {e}")
            raise
    
    def add_documents(self, documents):
        """Add documents to vector store
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        try:
            vector_store = self.get_vector_store()
            ids = vector_store.add_documents(documents)
            logger.info(f"Added {len(ids)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search_documents(self, query: str, k: int = 3):
        """Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            vector_store = self.get_vector_store()
            results = vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_info(self):
        """Get information about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            vector_store = self.get_vector_store()
            
            # Get collection from Chroma
            collection = vector_store._collection
            count = collection.count()
            
            return {
                "name": self.collection_name,
                "points_count": count,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "points_count": 0,
                "error": str(e)
            }
    
    def delete_collection(self):
        """Delete the collection and all its data"""
        try:
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"Deleted collection at: {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)"""
        try:
            self.delete_collection()
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise