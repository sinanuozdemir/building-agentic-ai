import json
import time
import hmac
import hashlib
import requests
import threading
import ssl
import websocket
import os
from typing import Dict, Any, Optional, Tuple
import base64
from dotenv import dotenv_values

class PusherBrowserClient:
    """Client for the Pusher-based Chrome extension browser automation."""
    
    def __init__(self, 
                 pusher_app_id: str = None,
                 pusher_app_key: str = None, 
                 pusher_app_secret: str = None,
                 pusher_cluster: str = "us3",
                 pusher_channel: str = "my-channel",
                 pusher_event: str = "my-event",
                 response_channel: str = "response-channel",
                 response_event: str = "response-event"):
        """Initialize the Pusher browser client.
        
        Args:
            pusher_app_id: Pusher App ID
            pusher_app_key: Pusher App Key
            pusher_app_secret: Pusher App Secret
            pusher_cluster: Pusher cluster (default: us3)
            pusher_channel: Channel to send commands to
            pusher_event: Event name for commands
            response_channel: Channel to receive responses
            response_event: Event name for responses
        """
        # Load config from .env file if not provided
        if not all([pusher_app_id, pusher_app_key, pusher_app_secret]):
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            env_vars = dotenv_values(env_path)
            
            self.pusher_app_id = pusher_app_id or env_vars.get("PUSHER_APP_ID")
            self.pusher_app_key = pusher_app_key or env_vars.get("PUSHER_APP_KEY")
            self.pusher_app_secret = pusher_app_secret or env_vars.get("PUSHER_APP_SECRET")
        else:
            self.pusher_app_id = pusher_app_id
            self.pusher_app_key = pusher_app_key
            self.pusher_app_secret = pusher_app_secret
            
        self.pusher_cluster = pusher_cluster
        self.pusher_channel = pusher_channel
        self.pusher_event = pusher_event
        self.response_channel = response_channel
        self.response_event = response_event
        
        # Validate required fields
        if not all([self.pusher_app_id, self.pusher_app_key, self.pusher_app_secret]):
            raise ValueError("Pusher App ID, Key, and Secret are required")
            
        # Set up connection state
        self.ws = None
        self.connected = False
        self.received_results = None
        self.last_screenshot_path = None
    
    def health_check(self) -> bool:
        """Check if the Pusher service is accessible."""
        try:
            # Test the Pusher API with a simple authentication check
            timestamp = int(time.time())
            auth_version = "1.0"
            path = f"/apps/{self.pusher_app_id}/channels"
            query_string = f"auth_key={self.pusher_app_key}&auth_timestamp={timestamp}&auth_version={auth_version}"
            
            # Create the string to sign
            string_to_sign = f"GET\n{path}\n{query_string}"
            
            # Generate the signature
            signature = hmac.new(
                self.pusher_app_secret.encode(),
                string_to_sign.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Full URL with authentication parameters
            url = f"https://api-{self.pusher_cluster}.pusher.com{path}?{query_string}&auth_signature={signature}"
            
            # Make the request
            response = requests.get(url)
            return response.status_code == 200
        except Exception:
            return False
    
    def _send_pusher_event(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Send an event to Pusher.
        
        Args:
            payload: The payload to send
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # JSON encode the payload
            payload_json = json.dumps(payload)
            
            # Create the data for the request
            data = {
                "name": self.pusher_event,
                "channel": self.pusher_channel,
                "data": payload_json
            }
            
            # JSON encode the data
            data_json = json.dumps(data)
            
            # Calculate MD5 hash of the request body
            body_md5 = hashlib.md5(data_json.encode()).hexdigest()
            
            # Current timestamp
            timestamp = int(time.time())
            
            # Create authentication parameters
            auth_version = "1.0"
            path = f"/apps/{self.pusher_app_id}/events"
            query_string = f"auth_key={self.pusher_app_key}&auth_timestamp={timestamp}&auth_version={auth_version}&body_md5={body_md5}"
            
            # Create the string to sign
            string_to_sign = f"POST\n{path}\n{query_string}"
            
            # Generate the signature
            signature = hmac.new(
                self.pusher_app_secret.encode(),
                string_to_sign.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Full URL with authentication parameters
            url = f"https://api-{self.pusher_cluster}.pusher.com{path}?{query_string}&auth_signature={signature}"
            
            # Make the request
            response = requests.post(url, data=data_json, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                return True, "Event sent successfully"
            else:
                return False, f"Failed to send event: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Error sending Pusher event: {str(e)}"
    
    def _listen_for_results(self, request_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Listen for results on the Pusher response channel.
        
        Args:
            request_id: The request ID to match responses
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Response data or None if timeout
        """
        self.received_results = None
        
        # WebSocket event handlers
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                if data.get('event') == self.response_event:
                    # Parse the result data
                    result_data = json.loads(data.get('data', '{}'))
                    
                    # Check if this is our request
                    if result_data.get('requestId') == request_id:
                        print(f"Received execution results for request {request_id}")
                        self.received_results = result_data
                        ws.close()
                
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_open(ws):
            print("WebSocket connection established to Pusher")
            
            # Subscribe to the response channel
            subscribe_msg = {
                "event": "pusher:subscribe",
                "data": {
                    "channel": self.response_channel
                }
            }
            ws.send(json.dumps(subscribe_msg))
        
        # Create WebSocket connection to Pusher
        socket_url = f"wss://ws-{self.pusher_cluster}.pusher.com/app/{self.pusher_app_key}?protocol=7&client=python&version=0.1.0"
        ws = websocket.WebSocketApp(
            socket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in a background thread
        ws_thread = threading.Thread(target=ws.run_forever, kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}})
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for results or timeout
        start_time = time.time()
        try:
            while time.time() - start_time < timeout and not self.received_results:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        # Close the WebSocket if it's still running
        if ws.sock and ws.sock.connected:
            ws.close()
        
        return self.received_results
    
    def init_browser(self, session_id: str = "default", connect_to_cdp: str = None,
                    browser_type: str = "chrome_extension") -> Tuple[bool, str]:
        """Initialize a browser instance.
        
        For Chrome extension, this is a no-op as the browser is already running.
        This method exists for compatibility with BrowserServiceClient.
        
        Args:
            session_id: Unique identifier for the browser session (ignored)
            connect_to_cdp: Optional Chrome DevTools Protocol endpoint (ignored)
            browser_type: Type of browser (should be "chrome_extension")
            
        Returns:
            Tuple of (success, message)
        """
        if browser_type != "chrome_extension":
            return False, "Only chrome_extension browser type is supported"
            
        # Check if Pusher is accessible
        if not self.health_check():
            return False, "Unable to connect to Pusher API"
            
        return True, "Chrome extension mode ready. Browser is controlled by extension."
    
    def execute_code(self, code: str, session_id: str = "default") -> Tuple[bool, str, Optional[str]]:
        """Execute code in the browser via Pusher.
        
        Args:
            code: The Playwright code to execute
            session_id: Unique identifier for the browser session (used as part of request ID)
            
        Returns:
            Tuple of (success, output, screenshot_path)
        """
        # Create unique request ID
        request_id = f"{session_id}-{int(time.time())}"
        
        # Prepare for receiving results
        self.received_results = None
        
        # Create the payload with response channel information
        payload = {
            "action": "executePlaywrightCode",
            "code": code.strip(),
            "requestId": request_id,
            "responseChannel": self.response_channel,
            "responseEvent": self.response_event
        }
        
        # Start listening for results in the background before sending
        # This avoids missing the response if it comes very quickly
        listen_thread = threading.Thread(target=self._listen_for_results, args=(request_id,))
        listen_thread.daemon = True
        listen_thread.start()
        
        # Send the payload
        success, message = self._send_pusher_event(payload)
        
        if not success:
            return False, message, None
        
        # Wait for the listener thread to complete
        listen_thread.join(30)  # Wait up to 30 seconds
        
        # Process results
        if self.received_results:
            result = self.received_results.get('result', '')
            logs = self.received_results.get('logs', [])
            
            # Format logs into output
            output_lines = []
            for log_type, log_message in logs:
                prefix = "LOG: " if log_type == "log" else f"{log_type.upper()}: "
                output_lines.append(f"{prefix}{log_message}")
            
            # Add result to output
            if result is not None:
                if isinstance(result, dict) or isinstance(result, list):
                    output_lines.append(f"RESULT: {json.dumps(result, indent=2)}")
                else:
                    output_lines.append(f"RESULT: {result}")
            
            # Check for screenshot in the code
            screenshot_path = None
            if 'take_screenshot = True' in code:
                screenshot_path = self._process_screenshot(session_id, code)
            
            return True, "\n".join(output_lines), screenshot_path
        else:
            return False, "Timeout waiting for execution results", None
    
    def _process_screenshot(self, session_id: str, code: str) -> Optional[str]:
        """Process a screenshot request embedded in the code.
        
        Args:
            session_id: Unique identifier for the browser session
            code: The code that was executed
            
        Returns:
            Path to the screenshot file, or None if no screenshot was taken
        """
        # In normal operation, a screenshot would be included in the results
        # Since the Chrome extension doesn't support this directly, we have to 
        # send a separate command to take a screenshot.
        
        # Create unique request ID for screenshot
        request_id = f"{session_id}-screenshot-{int(time.time())}"
        
        # Create the payload for taking a screenshot
        payload = {
            "action": "executePlaywrightCode",
            "code": "return await page.screenshot({ type: 'png', fullPage: true });",
            "requestId": request_id,
            "responseChannel": self.response_channel,
            "responseEvent": self.response_event
        }
        
        # Send the request and wait for results
        success, _ = self._send_pusher_event(payload)
        if not success:
            return None
            
        results = self._listen_for_results(request_id)
        
        if results and results.get('result'):
            try:
                # The result should be a base64-encoded PNG
                screenshot_data = results['result']
                if isinstance(screenshot_data, str) and screenshot_data.startswith('data:image/png;base64,'):
                    # Extract the base64 part
                    base64_data = screenshot_data.split(',')[1]
                else:
                    # Assume it's already base64
                    base64_data = screenshot_data
                    
                # Save to file
                screenshot_path = f"screenshot_{session_id}.png"
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(base64_data))
                return screenshot_path
            except Exception as e:
                print(f"Error processing screenshot: {e}")
                return None
        
        return None
    
    def get_browser_status(self, session_id: str = "default") -> Tuple[bool, str]:
        """Get the status of the browser.
        
        Args:
            session_id: Unique identifier for the browser session
            
        Returns:
            Tuple of (success, status message)
        """
        # Send a simple status check command
        request_id = f"{session_id}-status-{int(time.time())}"
        
        payload = {
            "action": "executePlaywrightCode",
            "code": "return { loaded: true };",
            "requestId": request_id,
            "responseChannel": self.response_channel,
            "responseEvent": self.response_event
        }
        
        # Send the request and wait for results
        success, message = self._send_pusher_event(payload)
        if not success:
            return False, f"Failed to check browser status: {message}"
            
        results = self._listen_for_results(request_id)
        
        if results and results.get('result'):
            loaded = results['result'].get('loaded', False)
            if loaded:
                return True, "Browser is ready"
            else:
                return False, "Browser is not ready"
        else:
            return False, "Failed to get browser status"
     
    def close_browser(self, session_id: str = "default") -> Tuple[bool, str]:
        """Close a browser instance.
        
        For Chrome extension, this is a no-op as we don't control the browser lifecycle.
        
        Args:
            session_id: Unique identifier for the browser session
            
        Returns:
            Tuple of (success, message)
        """
        return True, "Chrome extension mode doesn't close the browser" 