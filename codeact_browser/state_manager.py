import json
import os
from typing import Dict, List, Any
import streamlit as st

class StateManager:
    """Manages serialization and deserialization of Streamlit session state."""
    
    def __init__(self, state_dir: str = "chat_states"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
    
    def get_state_file(self, session_id: str) -> str:
        """Get the path to the state file for a given session."""
        return os.path.join(self.state_dir, f"{session_id}.json")
    
    def save_state(self, session_id: str) -> bool:
        """Save the current session state to a file."""
        try:
            # Extract only serializable portions of agent context
            browser_context = {}
            if "agent" in st.session_state and hasattr(st.session_state.agent, "context"):
                # Don't try to serialize browser-related objects
                context = st.session_state.agent.context
                for key, value in context.items():
                    # Skip browser-related objects that can't be serialized
                    if key not in ["browser_instance", "playwright_instance", "browser_page"] and not key.startswith("__"):
                        # Only serialize primitive types
                        if isinstance(value, (str, int, float, bool, dict, list, type(None))):
                            browser_context[key] = value
                            
                # Store browser state flags
                browser_context["had_browser"] = "browser_instance" in context
                browser_context["had_page"] = "browser_page" in context
                browser_context["browser_ready"] = context.get("browser_ready", False)
                
                # Store current URL if available
                try:
                    if "browser_page" in context and context["browser_page"] is not None:
                        browser_context["last_url"] = context["browser_page"].url()
                except Exception as e:
                    print(f"Error getting page URL: {str(e)}")
            
            state = {
                "messages": st.session_state.messages,
                "api_key": st.session_state.api_key,
                "browser_context": browser_context,
                "current_session": session_id
            }
            
            with open(self.get_state_file(session_id), "w") as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving state: {str(e)}")
            return False
    
    def load_state(self, session_id: str) -> bool:
        """Load a saved session state from a file."""
        try:
            state_file = self.get_state_file(session_id)
            if not os.path.exists(state_file):
                return False
                
            with open(state_file, "r") as f:
                state = json.load(f)
            
            # Load messages and API key
            st.session_state.messages = state["messages"]
            st.session_state.api_key = state["api_key"]
            
            # Set current session
            st.session_state.current_session = state.get("current_session", session_id)
            
            # We'll initialize the agent in the main file after loading
            # but we can store the browser context for later use
            st.session_state.browser_context = state.get("browser_context", {})
            
            return True
        except Exception as e:
            st.error(f"Error loading state: {str(e)}")
            return False
    
    def list_saved_states(self) -> List[str]:
        """List all saved session states."""
        try:
            return [f.replace(".json", "") for f in os.listdir(self.state_dir) 
                   if f.endswith(".json")]
        except Exception:
            return []
    
    def delete_state(self, session_id: str) -> bool:
        """Delete a saved session state."""
        try:
            state_file = self.get_state_file(session_id)
            if os.path.exists(state_file):
                os.remove(state_file)
                return True
            return False
        except Exception as e:
            st.error(f"Error deleting state: {str(e)}")
            return False 