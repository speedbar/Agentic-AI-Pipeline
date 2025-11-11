from langsmith import Client
from langsmith.schemas import Example, Run
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LangSmithEvaluator:
    """Evaluates LLM responses using LangSmith"""
    
    def __init__(self):
        try:
            self.client = Client()
            logger.info("LangSmith client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.client = None
    
    def accuracy_evaluator(self, run: Run, example: Example) -> dict:
        """Evaluate response accuracy
        
        Args:
            run: The LangSmith run containing outputs
            example: The example containing expected outputs
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get the response from run outputs
            response = run.outputs.get("response", "")
            expected = example.outputs.get("expected_response", "")
            
            # Simple exact match for now (can be enhanced)
            score = 1.0 if response.strip() == expected.strip() else 0.0
            
            return {
                "key": "accuracy",
                "score": score,
                "comment": f"Exact match: {score == 1.0}"
            }
            
        except Exception as e:
            logger.error(f"Accuracy evaluation error: {e}")
            return {"key": "accuracy", "score": 0.0, "comment": str(e)}
    
    def relevance_evaluator(self, run: Run, example: Example) -> dict:
        """Evaluate response relevance using LLM
        
        Args:
            run: The LangSmith run containing outputs
            example: The example containing the query
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            
            query = example.inputs.get("query", "")
            response = run.outputs.get("response", "")
            
            # Create evaluation prompt
            relevance_prompt = f"""Evaluate if the response is relevant to the query on a scale of 0-1.
            
Query: {query}

Response: {response}

Consider:
1. Does the response address the query?
2. Is the information accurate and helpful?
3. Is the response appropriate in scope?

Return ONLY a decimal number between 0 and 1, where:
- 1.0 = Highly relevant and helpful
- 0.5 = Partially relevant
- 0.0 = Not relevant

Score:"""
            
            # Get LLM evaluation
            result = llm.invoke(relevance_prompt)
            score_text = result.content.strip()
            
            # Parse score
            try:
                score = float(score_text)
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                logger.warning(f"Could not parse score: {score_text}")
                score = 0.5
            
            return {
                "key": "relevance",
                "score": score,
                "comment": f"LLM evaluated relevance: {score}"
            }
            
        except Exception as e:
            logger.error(f"Relevance evaluation error: {e}")
            return {"key": "relevance", "score": 0.5, "comment": str(e)}
    
    def completeness_evaluator(self, run: Run, example: Example) -> dict:
        """Evaluate if response is complete
        
        Args:
            run: The LangSmith run containing outputs
            example: The example containing the query
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            response = run.outputs.get("response", "")
            
            # Simple heuristics for completeness
            min_length = 50  # Minimum character count
            has_error = "error" in response.lower()
            
            if has_error:
                score = 0.0
            elif len(response) < min_length:
                score = 0.3
            else:
                score = 1.0
            
            return {
                "key": "completeness",
                "score": score,
                "comment": f"Response length: {len(response)}, Has error: {has_error}"
            }
            
        except Exception as e:
            logger.error(f"Completeness evaluation error: {e}")
            return {"key": "completeness", "score": 0.0, "comment": str(e)}
    
    def evaluate_response(self, query: str, response: str, route: str) -> dict:
        """Evaluate a single response
        
        Args:
            query: The user query
            response: The generated response
            route: The route taken (weather or documents)
            
        Returns:
            Dictionary with evaluation scores
        """
        scores = {
            "query": query,
            "response": response,
            "route": route,
            "scores": {}
        }
        
        try:
            # Length check
            scores["scores"]["length"] = len(response)
            
            # Route appropriateness
            if route == "weather":
                weather_terms = ["temperature", "weather", "degrees", "humidity", "wind"]
                has_weather_terms = any(term in response.lower() for term in weather_terms)
                scores["scores"]["route_appropriate"] = 1.0 if has_weather_terms else 0.5
            else:
                scores["scores"]["route_appropriate"] = 0.8  # Harder to evaluate for documents
            
            # Error detection
            has_error = "error" in response.lower() or len(response) < 20
            scores["scores"]["success"] = 0.0 if has_error else 1.0
            
            # Overall score
            overall = sum(scores["scores"].values()) / len(scores["scores"])
            scores["overall_score"] = overall
            
            logger.info(f"Evaluation complete. Overall score: {overall}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            scores["error"] = str(e)
        
        return scores
    
    def log_to_langsmith(self, query: str, response: str, route: str, metadata: dict = None):
        """Log evaluation results to LangSmith
        
        Args:
            query: The user query
            response: The generated response
            route: The route taken
            metadata: Additional metadata to log
        """
        if not self.client:
            logger.warning("LangSmith client not available, skipping logging")
            return
        
        try:
            # Evaluate the response
            eval_results = self.evaluate_response(query, response, route)
            
            # Add metadata
            if metadata:
                eval_results.update(metadata)
            
            # Log to LangSmith
            logger.info(f"LangSmith evaluation: {eval_results}")
            
            # You can also create a dataset and add examples here
            # self.client.create_example(...)
            
        except Exception as e:
            logger.error(f"Failed to log to LangSmith: {e}")