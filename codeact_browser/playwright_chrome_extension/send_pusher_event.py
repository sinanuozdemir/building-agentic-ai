#!/usr/bin/env python3
import json
import time
import hmac
import hashlib
import urllib.parse
import requests
import sys
import websocket
import threading
import ssl

# Import config
try:
    from config import PUSHER_CONFIG, CHANNEL, EVENT, RESPONSE_CHANNEL, RESPONSE_EVENT
except ImportError:
    print("Error: config.py not found. Please copy config.example.py to config.py and update with your Pusher credentials.")
    sys.exit(1)

# User's specific code
USER_CODE = """
// Go to my site
await page.goto('https://sinanozdemir.ai');

// Click the first link on the page (likely the first nav/link)
await page.waitForSelector('a');
await page.click('a');

// Give the new page a moment to load and capture its title
await page.waitForSelector('title');

return {finalURL: window.location.href, finalTitle: document.title};
"""

# Global to store received results
received_results = None

def pretty_print_json(data):
    """Format and print JSON data with indentation"""
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            print(json.dumps(parsed, indent=2))
        except:
            print(data)
    else:
        print(json.dumps(data, indent=2))

def listen_for_results(request_id, timeout=30):
    """Listen for results on the Pusher response channel"""
    global received_results
    
    print(f"\nListening for results on {RESPONSE_CHANNEL}:{RESPONSE_EVENT}")
    print(f"Request ID: {request_id}")
    print(f"Timeout: {timeout} seconds")
    
    # WebSocket event handlers
    def on_message(ws, message):
        global received_results
        try:
            data = json.loads(message)
            
            if data.get('event') == RESPONSE_EVENT:
                # Parse the result data
                result_data = json.loads(data.get('data', '{}'))
                
                # Check if this is our request
                if result_data.get('requestId') == request_id:
                    print("\n✅ Received execution results:")
                    pretty_print_json(result_data)
                    received_results = result_data
                    ws.close()
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("WebSocket connection closed")
    
    def on_open(ws):
        print("WebSocket connection established")
        
        # Subscribe to the response channel
        subscribe_msg = {
            "event": "pusher:subscribe",
            "data": {
                "channel": RESPONSE_CHANNEL
            }
        }
        ws.send(json.dumps(subscribe_msg))
    
    # Create WebSocket connection to Pusher
    socket_url = f"wss://ws-{PUSHER_CONFIG['CLUSTER']}.pusher.com/app/{PUSHER_CONFIG['APP_KEY']}?protocol=7&client=python&version=0.1.0"
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
        while time.time() - start_time < timeout and not received_results:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    if not received_results:
        print("\n❌ Timed out waiting for results")
    
    # Close the WebSocket if it's still running
    if ws.sock and ws.sock.connected:
        ws.close()
    
    return received_results

def send_pusher_event(code=USER_CODE, url=None):
    # Create unique request ID
    request_id = f"test-{int(time.time())}"
    
    # Create the payload with response channel information
    payload = {
        "action": "executePlaywrightCode",
        "code": code.strip(),
        "requestId": request_id,
        "responseChannel": RESPONSE_CHANNEL,
        "responseEvent": RESPONSE_EVENT
    }
    
    # If URL is provided, override the URL in the code
    if url:
        # Replace the URL in the code
        modified_code = code.replace("https://sinanozdemir.ai", url)
        payload["code"] = modified_code
    
    # JSON encode the payload
    payload_json = json.dumps(payload)
    
    # Create the data for the request
    data = {
        "name": EVENT,
        "channel": CHANNEL,
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
    path = f"/apps/{PUSHER_CONFIG['APP_ID']}/events"
    query_string = f"auth_key={PUSHER_CONFIG['APP_KEY']}&auth_timestamp={timestamp}&auth_version={auth_version}&body_md5={body_md5}"
    
    # Create the string to sign
    string_to_sign = f"POST\n{path}\n{query_string}"
    
    # Generate the signature
    signature = hmac.new(
        PUSHER_CONFIG['APP_SECRET'].encode(),
        string_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Full URL with authentication parameters
    url = f"https://api-{PUSHER_CONFIG['CLUSTER']}.pusher.com{path}?{query_string}&auth_signature={signature}"
    
    # Make the request
    print(f"Sending Pusher event to {CHANNEL}:{EVENT}")
    print("\nCode being executed:")
    print("-------------------")
    print(payload["code"])
    print("-------------------")
    
    print("\nSending payload:")
    pretty_print_json(payload)
    
    # Start listening for results in the background before sending
    # This avoids missing the response if it comes very quickly
    listen_thread = threading.Thread(target=listen_for_results, args=(request_id,))
    listen_thread.daemon = True
    listen_thread.start()
    
    # Send the event
    response = requests.post(url, data=data_json, headers={"Content-Type": "application/json"})
    
    print("\nPusher API Response:")
    if response.status_code == 200:
        print("✅ Event sent successfully!")
        try:
            response_json = response.json()
            pretty_print_json(response_json)
        except:
            print(response.text)
        
        # Wait for the listener thread to complete
        listen_thread.join(30)  # Wait up to 30 seconds
        
        if received_results:
            print("\nFinal result captured:")
            pretty_print_json(received_results.get('result', {}))
        else:
            print("\nNo results received from extension.")
            print("Make sure the extension is running and properly configured to send results.")
    else:
        print(f"❌ Error sending event: {response.status_code}")
        try:
            response_json = response.json()
            pretty_print_json(response_json)
        except:
            print(response.text)

if __name__ == "__main__":
    try:
        # Install websocket-client if needed
        import pkg_resources
        pkg_resources.require('websocket-client')
    except (ImportError, pkg_resources.DistributionNotFound):
        print("Installing required package: websocket-client")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client"])
        print("Package installed successfully.")
    
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"Navigating to URL: {url}")
        send_pusher_event(url=url)
    else:
        send_pusher_event() 