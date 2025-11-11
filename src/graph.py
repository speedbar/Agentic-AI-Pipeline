from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from typing import TypedDict, List, Any
from llm_provider import get_azure_llm
from vector_db import VectorDatabaseManager
from weather_api import WeatherAPI
import logging

logger = logging.getLogger(__name__)


@tool
def weather_tool(location: str) -> str:
    """Get weather info for a location.
    
    Args:
        location: The city or location name to get weather for
        
    Returns:
        Weather information as a string
    """
    from weather_api import WeatherAPI
    weather_api = WeatherAPI()
    result = weather_api.get_current_weather(location)
    return str(result)
    
@tool
def document_tool(query: str) -> str:
    """Retrieve relevant documents for the query.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        Concatenated relevant document content
    """
    from vector_db import VectorDatabaseManager
    vector_db = VectorDatabaseManager()
    vector_store = vector_db.get_vector_store()
    results = vector_store.similarity_search(query, k=3)
    if results:
        return "\n---\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in results
        )
    return "No relevant documents found."


class AgentState(TypedDict):
    query: str
    route: str
    weather_data: str
    document_data: str
    response: str
    messages: List[Any]


class RagAgentGraph:
    def __init__(self):
        self.llm = get_azure_llm(temperature=0.7)
        self.vector_db = VectorDatabaseManager()
        self.weather_api = WeatherAPI()
        self.graph = None

    def route_node(self, state: AgentState) -> AgentState:
        """Decides which path to follow: weather or document"""
        weather_keywords = ["weather", "temperature", "forecast", "wind", "rain", 
                           "humidity", "snow", "climate", "sunny", "cloudy"]
        query = state["query"].lower()
        
        if any(word in query for word in weather_keywords):
            state["route"] = "weather"
            logger.info(f"Routed to weather for query: {state['query']}")
        else:
            state["route"] = "documents"
            logger.info(f"Routed to documents for query: {state['query']}")
        
        return state

    def weather_node(self, state: AgentState) -> AgentState:
        """Fetch weather data for the query"""
        try:
            # Extract location from query
            import re
            pattern = r"(?:in|for|at)\s+([A-Za-z\s]+?)(?:\s|$|\?|,)"
            m = re.search(pattern, state["query"], re.IGNORECASE)
            location = m.group(1).strip() if m else "London"
            
            logger.info(f"Fetching weather for location: {location}")
            
            # Call weather tool
            weather_data = weather_tool.invoke({"location": location})
            state["weather_data"] = str(weather_data)
            logger.info(f"Weather data retrieved: {state['weather_data'][:100]}...")
            
        except Exception as e:
            logger.error(f"Error in weather node: {e}")
            state["weather_data"] = f"Error fetching weather: {str(e)}"
        
        return state

    def document_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents for the query"""
        try:
            logger.info(f"Retrieving documents for query: {state['query']}")
            
            # Call document tool
            doc_data = document_tool.invoke({"query": state["query"]})
            state["document_data"] = doc_data
            logger.info(f"Document data retrieved: {len(state['document_data'])} characters")
            
        except Exception as e:
            logger.error(f"Error in document node: {e}")
            state["document_data"] = f"Error retrieving documents: {str(e)}"
        
        return state

    def response_node(self, state: AgentState) -> AgentState:
        """Generate final response using LLM"""
        try:
            # Prepare prompt based on route
            if state["route"] == "weather":
                context = f"Weather data: {state['weather_data']}"
                instruction = "Based on the weather data above, provide a helpful and natural response."
            else:
                context = f"Document data: {state['document_data']}"
                instruction = "Based on the document data above, answer the user's question accurately."
            
            prompt = f"""User query: {state['query']}

Context:
{context}

{instruction}

Please provide a clear, concise, and helpful response."""

            logger.info("Generating response with LLM...")
            response = self.llm.invoke(prompt)
            state["response"] = getattr(response, "content", str(response))
            logger.info(f"Response generated: {len(state['response'])} characters")
            
        except Exception as e:
            logger.error(f"Error in response node: {e}")
            state["response"] = f"Error generating response: {str(e)}"
        
        return state

    def build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("route_node", self.route_node)
        builder.add_node("weather_node", self.weather_node)
        builder.add_node("document_node", self.document_node)
        builder.add_node("response_node", self.response_node)

        # Add edges
        builder.add_edge(START, "route_node")

        # Conditional edge based on routing decision
        def which_node(state: AgentState) -> str:
            return "weather_node" if state["route"] == "weather" else "document_node"

        builder.add_conditional_edges("route_node", which_node)
        builder.add_edge("weather_node", "response_node")
        builder.add_edge("document_node", "response_node")
        builder.add_edge("response_node", END)
        
        self.graph = builder.compile()
        logger.info("Graph built successfully")

    def invoke(self, query: str):
        """Execute the graph with a query"""
        if not self.graph:
            self.build_graph()
        
        initial_state = {
            "query": query,
            "route": "",
            "weather_data": "",
            "document_data": "",
            "response": "",
            "messages": [],
        }
        
        logger.info(f"Invoking graph with query: {query}")
        result = self.graph.invoke(initial_state)
        return result