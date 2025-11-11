import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def listapi():
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    return api_key, endpoint, api_version, deployment_name, embedding_deployment


class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")


class VectorDBConfig:
    """Vector Database configuration - Using ChromaDB (local, no Docker needed)"""
    persist_directory: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    collection_name: str = "documents"


class WeatherConfig:
    """OpenWeatherMap API configuration"""
    api_key: str = os.getenv("OPENWEATHERMAP_API_KEY")


class LangSmithConfig:
    """LangSmith configuration"""
    api_key: str = os.getenv("LANGSMITH_API_KEY")
    tracing: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    project: str = os.getenv("LANGSMITH_PROJECT", "ai-engineer-assignment")