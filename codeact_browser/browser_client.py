import requests
import base64
import json
from typing import Dict, Any, Optional, Tuple
import os

class BrowserServiceClient:
    """Client for the browser service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the browser service client.
        
        Args:
            base_url: Base URL of the browser service
        """
        self.base_url = base_url
        
    def health_check(self) -> bool:
        """Check if the browser service is running."""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception:
            return False
    
    def init_browser(self, session_id: str = "default", connect_to_cdp: str = None,
                    browser_type: str = "default", anchor_api_key: str = None) -> Tuple[bool, str]:
        """Initialize a browser instance.
        
        Args:
            session_id: Unique identifier for the browser session
            connect_to_cdp: Optional Chrome DevTools Protocol endpoint to connect to an existing browser
            browser_type: Type of browser to use ("default", "existing", "anchor")
            anchor_api_key: API key for Anchor Browser (required if browser_type is "anchor")
            
        Returns:
            Tuple of (success, message)
        """
        try:
            params = {"session_id": session_id}
            if connect_to_cdp:
                params["connect_to_cdp"] = connect_to_cdp
            if browser_type:
                params["browser_type"] = browser_type
            if anchor_api_key:
                params["anchor_api_key"] = anchor_api_key
            
            response = requests.post(
                f"{self.base_url}/browser/init",
                params=params
            )
            data = response.json()
            return data["success"], data.get("output", data.get("error", "Unknown error"))
        except Exception as e:
            return False, f"Error communicating with browser service: {str(e)}"
    
    def execute_code(self, code: str, session_id: str = "default") -> Tuple[bool, str, Optional[str]]:
        """Execute code in the browser.
        
        Args:
            code: The Playwright code to execute
            session_id: Unique identifier for the browser session
            
        Returns:
            Tuple of (success, output, screenshot_path)
        """
        try:
            response = requests.post(
                f"{self.base_url}/browser/execute",
                json={"code": code, "session_id": session_id}
            )
            data = response.json()
            
            # Handle screenshot if present
            screenshot_path = None
            if data.get("screenshot"):
                # Save screenshot to a file
                img_data = base64.b64decode(data["screenshot"])
                screenshot_path = f"screenshot_{session_id}.png"
                with open(screenshot_path, "wb") as f:
                    f.write(img_data)
            
            return (
                data["success"], 
                data.get("output", data.get("error", "Unknown error")),
                screenshot_path
            )
        except Exception as e:
            return False, f"Error communicating with browser service: {str(e)}", None
    
    def get_browser_status(self, session_id: str = "default") -> Tuple[bool, str]:
        """Get the status of the browser.
        
        Args:
            session_id: Unique identifier for the browser session
            
        Returns:
            Tuple of (success, status message)
        """
        try:
            response = requests.get(
                f"{self.base_url}/browser/status",
                params={"session_id": session_id}
            )
            data = response.json()
            return data["success"], data.get("output", data.get("error", "Unknown error"))
        except Exception as e:
            return False, f"Error communicating with browser service: {str(e)}"
    
    def take_screenshot(self, session_id: str = "default") -> Tuple[bool, str, Optional[str]]:
        """Take a screenshot of the current page.
        
        Args:
            session_id: Unique identifier for the browser session
            
        Returns:
            Tuple of (success, message, screenshot_path)
        """
        try:
            response = requests.post(
                f"{self.base_url}/browser/screenshot",
                params={"session_id": session_id}
            )
            data = response.json()
            
            # Handle screenshot if present
            screenshot_path = None
            if data.get("screenshot"):
                # Save screenshot to a file
                img_data = base64.b64decode(data["screenshot"])
                screenshot_path = f"screenshot_{session_id}.png"
                with open(screenshot_path, "wb") as f:
                    f.write(img_data)
            
            return (
                data["success"], 
                data.get("output", data.get("error", "Unknown error")),
                screenshot_path
            )
        except Exception as e:
            return False, f"Error communicating with browser service: {str(e)}", None
    
    def close_browser(self, session_id: str = "default") -> Tuple[bool, str]:
        """Close a browser instance.
        
        Args:
            session_id: Unique identifier for the browser session
            
        Returns:
            Tuple of (success, message)
        """
        try:
            response = requests.post(
                f"{self.base_url}/browser/close",
                params={"session_id": session_id}
            )
            data = response.json()
            return data["success"], data.get("output", data.get("error", "Unknown error"))
        except Exception as e:
            return False, f"Error communicating with browser service: {str(e)}" 