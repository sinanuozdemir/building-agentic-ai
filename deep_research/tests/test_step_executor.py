import unittest
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from step_executor import search_serp_tool, tools, agent_executor

class TestStepExecutor(unittest.TestCase):
    
    def test_search_tool_exists(self):
        """Test that the search tool is initialized correctly."""
        self.assertIsNotNone(search_serp_tool)
    
    def test_tools_list(self):
        """Test that tools list contains search tool."""
        self.assertIn(search_serp_tool, tools)
    
    def test_agent_executor(self):
        """Test that agent executor is initialized."""
        self.assertIsNotNone(agent_executor)
    
    def test_real_search(self):
        """Test that the search tool can perform a real search."""
        # Use the search tool to search for weather in San Francisco
        query = "weather in San Francisco today"
        print(f"\n--- Starting search for: '{query}' ---")
        
        result = search_serp_tool.run(query)
        
        # The result should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Check that the result contains relevant information
        relevant_keywords = ["san francisco", "weather", "temperature", "forecast", "°"]
        found_keywords = [keyword for keyword in relevant_keywords if keyword in result.lower()]
        
        self.assertTrue(len(found_keywords) > 0, 
                       f"No relevant keywords found. Expected at least one of: {relevant_keywords}")
        
        # Log detailed results
        print(f"\n=== Search Results for '{query}' ===")
        print(f"Result length: {len(result)} characters")
        print(f"Found keywords: {found_keywords}")
        print(f"\nFull result:\n{result}")
        print("="*50)

if __name__ == "__main__":
    unittest.main() 