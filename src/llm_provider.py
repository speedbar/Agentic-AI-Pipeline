from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from config import AzureOpenAIConfig, LangSmithConfig
import os

def get_azure_llm(temperature: float = 0.7):
    # \"\"\"Initialize Azure OpenAI LLM with LangSmith tracing\"\"\"
    
    # Enable LangSmith tracing
    if LangSmithConfig.tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = LangSmithConfig.api_key
        os.environ["LANGSMITH_PROJECT"] = LangSmithConfig.project
    
    llm = AzureChatOpenAI(
        azure_endpoint=AzureOpenAIConfig.endpoint,
        azure_deployment=AzureOpenAIConfig.deployment_name,
        api_version=AzureOpenAIConfig.api_version,
        api_key=AzureOpenAIConfig.api_key,
        temperature=temperature,
        max_tokens=2048,
    )
    return llm

def get_embeddings():
    # \"\"\"Initialize Azure OpenAI embeddings\"\"\"
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AzureOpenAIConfig.endpoint,
        azure_deployment=AzureOpenAIConfig.embedding_deployment,
        api_version=AzureOpenAIConfig.api_version,
        api_key=AzureOpenAIConfig.api_key,
    )
    return embeddings

