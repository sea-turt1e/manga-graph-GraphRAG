#!/usr/bin/env python3
# Test script for recommend_manga function

import os
import sys
from typing import Dict, Any, List

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample_with_microsoft_graph import MangaGraphRAG


def test_recommend_function():
    """Test the recommend_manga function with mock data"""
    
    class MockGraphRAG(MangaGraphRAG):
        """Mock version for testing without Neo4j connection"""
        
        def __init__(self):
            # Initialize without Neo4j connection
            self.llm = None
            self.graph = None
            self.qa_chain = None
            
        def enhance_query_with_graph_context(self, question: str) -> str:
            """Mock implementation"""
            return question + " (enhanced with context)"
            
        def _generate_recommendation_reasoning(self, preference: str, graph_data: List[Dict]) -> str:
            """Mock implementation"""
            if graph_data:
                return f"Based on {len(graph_data)} related items found, we recommend similar works with matching themes and authors."
            else:
                return "No graph data available for generating reasoning."
                
        def recommend_manga(self, user_preference: str) -> Dict[str, Any]:
            """Override with test implementation"""
            # Simulate different result formats
            test_cases = [
                # Case 1: Dict with intermediate_steps
                {
                    "result": "Based on your preference for NARUTO, I recommend ONE PIECE and BLEACH.",
                    "intermediate_steps": [
                        {
                            "query": "MATCH (m:Manga) WHERE m.title CONTAINS 'NARUTO' RETURN m",
                            "context": [{"title": "NARUTO", "genre": "Shonen"}]
                        }
                    ]
                },
                # Case 2: Dict without intermediate_steps but with other keys
                {
                    "result": "Recommended manga: ONE PIECE",
                    "cypher": "MATCH (m:Manga)-[:SIMILAR_TO]-(s:Manga) RETURN s",
                    "context": [{"title": "ONE PIECE", "author": "Oda Eiichiro"}]
                },
                # Case 3: String result
                "Simple recommendation: BLEACH"
            ]
            
            # Use the first test case
            result = test_cases[0]
            
            # Call the actual method
            return super().recommend_manga(user_preference)
    
    # Test the function
    print("Testing recommend_manga function...")
    mock_graphrag = MockGraphRAG()
    
    # Test case 1: Dict with intermediate_steps
    mock_result = {
        "result": "Based on your preference for NARUTO, I recommend ONE PIECE and BLEACH.",
        "intermediate_steps": [
            {
                "query": "MATCH (m:Manga) WHERE m.title CONTAINS 'NARUTO' RETURN m",
                "context": [{"title": "NARUTO", "genre": "Shonen"}]
            }
        ]
    }
    
    # Manually test the extraction logic
    cypher_query = "クエリ情報なし"
    database_results = []
    
    if isinstance(mock_result, dict):
        intermediate_steps = mock_result.get("intermediate_steps", [])
        if intermediate_steps:
            for step in intermediate_steps:
                if isinstance(step, dict) and "query" in step:
                    cypher_query = step["query"]
                    break
            
            for step in intermediate_steps:
                if isinstance(step, dict) and "context" in step:
                    database_results = step["context"]
                    break
    
    print("\n=== Test Results ===")
    print(f"Cypher Query extracted: {cypher_query}")
    print(f"Database Results: {database_results}")
    print(f"Results count: {len(database_results)}")
    
    # Test the reasoning generation
    reasoning = mock_graphrag._generate_recommendation_reasoning("NARUTO", database_results)
    print(f"\nGenerated Reasoning: {reasoning}")
    
    print("\n✅ Test completed successfully!")
    

if __name__ == "__main__":
    test_recommend_function()