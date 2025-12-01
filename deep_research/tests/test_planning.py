import unittest
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from planning import Plan, planner, replanner
from langchain_openai import ChatOpenAI

class TestPlanning(unittest.TestCase):
    
    def test_planner(self):
        """Test that the planner generates a valid plan."""
        test_objective = "Write a one page report on the ODSC East 2025 conference"
        test_messages = [("human", f"Objective: {test_objective}")]
        
        plan = planner.invoke({"messages": test_messages})
        
        self.assertIsInstance(plan, Plan)
        self.assertTrue(len(plan.steps) > 0)
        self.assertIsInstance(plan.steps, list)
        for step in plan.steps:
            self.assertIsInstance(step, str)
    
    def test_replanner(self):
        """Test that the replanner updates a plan correctly."""
        original_plan = Plan(
            steps=[
                "Identify the most recent game played by the Baltimore Ravens.", 
                "Determine the outcome of that game (win or loss).", 
                "If the Ravens won, identify the starting quarterback (QB) for that game.", 
                "Research the hometown of that quarterback."
            ]
        )
        
        replan = replanner.invoke(
            {
                "input": "what is the hometown of the QB of the winner of the most recent Ravens game?",
                "plan": original_plan,
                "past_steps": [('Identify the most recent game played by the Baltimore Ravens.', "They played the Steelers and won 27-24")],
            }
        )
        
        self.assertTrue(hasattr(replan, 'action'))
        # Could be a Plan or a Response
        if hasattr(replan.action, 'steps'):
            self.assertIsInstance(replan.action.steps, list)
            self.assertTrue(len(replan.action.steps) > 0)
            self.assertTrue(len(replan.action.steps) < len(original_plan.steps))
        else:
            self.assertTrue(hasattr(replan.action, 'response'))

if __name__ == "__main__":
    unittest.main() 