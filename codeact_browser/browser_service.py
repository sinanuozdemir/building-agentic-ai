from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import base64
import os
import traceback
import threading
from contextlib import asynccontextmanager
import concurrent.futures
import asyncio
import io
import contextlib
import requests
import time

try:
    from playwright.sync_api import sync_playwright, Browser, Page
except ImportError:
    raise ImportError("Playwright not installed. Run: pip install playwright && python -m playwright install")

# Global variables to store browser instances
browser_instances: Dict[str, Browser] = {}
page_instances: Dict[str, Page] = {}
playwright_instance = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Store anchor browser session data
anchor_sessions: Dict[str, Dict[str, Any]] = {}

# Synchronous functions to be run in thread pool
def init_playwright_sync():
    """Initialize playwright in a separate thread."""
    global playwright_instance
    try:
        playwright_instance = sync_playwright().start()
        print("Playwright initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Playwright: {str(e)}")
        traceback.print_exc()
        return False

def fix_browser_variable_names(code):
    """Translate browser_page to page in the code, to match what the service expects."""
    # Print original code for debugging
    print(f"Original code:\n{code}")
    
    # Replace browser_page with page
    fixed_code = code.replace("browser_page.", "page.")
    fixed_code = fixed_code.replace("browser_page =", "page =")
    
    # Also handle cases where browser_page is used as a variable on its own
    fixed_code = fixed_code.replace(" browser_page ", " page ")
    fixed_code = fixed_code.replace("(browser_page)", "(page)")
    fixed_code = fixed_code.replace("[browser_page]", "[page]")
    fixed_code = fixed_code.replace("{browser_page}", "{page}")
    fixed_code = fixed_code.replace(",browser_page,", ",page,")
    fixed_code = fixed_code.replace(",browser_page)", ",page)")
    fixed_code = fixed_code.replace("(browser_page,", "(page,")
    
    # Be careful with general replacement as it might affect strings or comments
    # Do it as a last resort if specific replacements didn't catch everything
    if "browser_page" in fixed_code:
        print("Warning: 'browser_page' still found in code after replacements")
        fixed_code = fixed_code.replace("browser_page", "page")
    
    # Print translated code for debugging
    print(f"Translated code:\n{fixed_code}")
    
    return fixed_code

def init_browser_sync(session_id: str, connect_to_cdp: Optional[str] = None, browser_type: str = "default", anchor_api_key: Optional[str] = None):
    """Initialize browser synchronously.
    
    Args:
        session_id: Unique identifier for the browser session
        connect_to_cdp: Optional CDP endpoint URL to connect to an existing browser
        browser_type: Type of browser to use ("default", "existing", "anchor")
        anchor_api_key: API key for Anchor Browser (required if browser_type is "anchor")
    """
    global playwright_instance, browser_instances, page_instances, anchor_sessions
    
    try:
        if playwright_instance is None:
            # Try initializing playwright if it's not ready
            if not init_playwright_sync():
                return False, "Failed to initialize Playwright"
            
        if session_id not in browser_instances:
            print(f"Creating browser for session {session_id}")
            
            # For Anchor Browser
            if browser_type == "anchor":
                if not anchor_api_key:
                    return False, "Anchor API key is required for Anchor Browser"
                
                try:
                    # Create an Anchor browser session
                    print("Creating Anchor Browser session...")
                    response = requests.post(
                        "https://api.anchorbrowser.io/v1/sessions",
                        headers={
                            "anchor-api-key": anchor_api_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "browser": {
                                "headless": {"active": False}
                            }
                        }
                    )
                    
                    if response.status_code != 200:
                        return False, f"Failed to create Anchor Browser session: {response.text}"
                    
                    session_data = response.json()["data"]
                    print(f"Anchor session created: {session_data['id']}")
                    
                    # Store session data for later use
                    anchor_sessions[session_id] = session_data
                    
                    # Generate the CDP URL for connecting
                    cdp_url = f"wss://connect.anchorbrowser.io?apiKey={anchor_api_key}&sessionId={session_data['id']}"
                    print(f"Connecting to Anchor Browser at {cdp_url}")
                    
                    # Connect to the Anchor browser
                    browser_instances[session_id] = playwright_instance.chromium.connect_over_cdp(cdp_url)
                    
                    # Get or create a page
                    contexts = browser_instances[session_id].contexts
                    if contexts and contexts[0].pages:
                        page_instances[session_id] = contexts[0].pages[0]
                    else:
                        context = browser_instances[session_id].new_context()
                        page_instances[session_id] = context.new_page()
                    
                    return True, f"Connected to Anchor Browser. Live view: {session_data['live_view_url']}"
                
                except Exception as e:
                    traceback.print_exc()
                    return False, f"Error connecting to Anchor Browser: {str(e)}"
            
            # If a CDP endpoint is provided or existing browser type is selected
            elif connect_to_cdp or browser_type == "existing":
                try:
                    cdp_endpoint = connect_to_cdp
                    print(f"Connecting to existing browser at {cdp_endpoint}")
                    browser_instances[session_id] = playwright_instance.chromium.connect_over_cdp(cdp_endpoint)
                    
                    # Get the default context and page or create a new one
                    contexts = browser_instances[session_id].contexts
                    if contexts:
                        pages = contexts[0].pages
                        if pages:
                            page_instances[session_id] = pages[0]
                        else:
                            page_instances[session_id] = contexts[0].new_page()
                    else:
                        context = browser_instances[session_id].new_context()
                        page_instances[session_id] = context.new_page()
                        
                    return True, f"Connected to existing browser at {cdp_endpoint}"
                except Exception as e:
                    traceback.print_exc()
                    return False, f"Error connecting to existing browser: {str(e)}"
            else:
                # Launch a new browser instance
                browser_instances[session_id] = playwright_instance.chromium.launch(headless=False)
                page_instances[session_id] = browser_instances[session_id].new_page()
            
        return True, "Browser initialized successfully"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error initializing browser: {str(e)}"

def execute_code_sync(session_id: str, code: str):
    """Execute code synchronously."""
    global browser_instances, page_instances
    
    try:
        if session_id not in browser_instances or session_id not in page_instances:
            return False, f"No browser instance found for session {session_id}", None
            
        browser = browser_instances[session_id]
        page = page_instances[session_id]
        
        # Fix variable names (translate browser_page to page)
        code = fix_browser_variable_names(code)
        
        # Prepare execution environment
        locals_dict = {
            "browser": browser,
            "page": page,
            "output": "",
            "take_screenshot": False,
            "screenshot_path": "screenshot.png"
        }
        
        # Redirect stdout to capture print statements
        stdout_buffer = io.StringIO()
        
        # Execute the command
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, globals(), locals_dict)
                
            # Get print output
            captured_prints = stdout_buffer.getvalue()
            
            # Get direct output assignment
            direct_output = locals_dict.get("output", "")
            
            # Combine outputs
            if captured_prints and direct_output:
                # We have both prints and direct output
                output = f"{captured_prints}\n{direct_output}"
            elif captured_prints:
                # We have only prints
                output = captured_prints
            elif direct_output:
                # We have only direct output
                output = direct_output
            else:
                # No output
                output = "Command executed successfully"
                
            print(f"Captured print output: {captured_prints!r}")
            print(f"Direct output variable: {direct_output!r}")
            print(f"Combined output: {output!r}")
            
            # Check if screenshot was requested
            screenshot_b64 = None
            if locals_dict.get("take_screenshot", False):
                screenshot_path = locals_dict.get("screenshot_path", "screenshot.png")
                page.screenshot(path=screenshot_path)
                
                # Convert screenshot to base64
                with open(screenshot_path, "rb") as f:
                    screenshot_b64 = base64.b64encode(f.read()).decode("utf-8")
                
                # Remove the temporary file
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
                    
            return True, output, screenshot_b64
        except Exception as e:
            traceback.print_exc()
            return False, f"Error executing command: {str(e)}", None
    except Exception as e:
        traceback.print_exc()
        return False, f"Error accessing browser: {str(e)}", None

def get_status_sync(session_id: str):
    """Get browser status synchronously."""
    global browser_instances, page_instances
    
    try:
        if session_id not in browser_instances or session_id not in page_instances:
            return False, "No browser instance found for this session"
            
        page = page_instances[session_id]
        url = page.url
        title = page.title()
        return True, f"Browser is running. Current page: {title} ({url})"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error accessing browser: {str(e)}"

def close_browser_sync(session_id: str):
    """Close browser synchronously."""
    global browser_instances, page_instances
    
    try:
        if session_id not in browser_instances:
            return False, f"No browser instance found for session {session_id}"
            
        browser_instances[session_id].close()
        del browser_instances[session_id]
        del page_instances[session_id]
        return True, f"Browser for session {session_id} closed successfully"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error closing browser: {str(e)}"

def shutdown_browser_sync():
    """Shutdown all browser instances synchronously."""
    global playwright_instance, browser_instances, page_instances
    
    try:
        for session_id, browser in list(browser_instances.items()):
            try:
                browser.close()
            except:
                pass
            
        browser_instances.clear()
        page_instances.clear()
        
        if playwright_instance:
            playwright_instance.stop()
            playwright_instance = None
            
        return True, "All browsers closed successfully"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error shutting down browsers: {str(e)}"

# Async wrapper functions
async def run_in_executor(func, *args):
    """Run a function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize playwright in a separate thread
    success = await run_in_executor(init_playwright_sync)
    if success:
        print("Playwright initialized successfully in lifespan")
    else:
        print("WARNING: Failed to initialize Playwright in lifespan")
    
    # Continue with app startup
    yield
    
    # Shutdown: close all browsers and stop playwright
    success, message = await run_in_executor(shutdown_browser_sync)
    print(f"Browser service shutdown: {success}, {message}")

app = FastAPI(title="Browser Service", lifespan=lifespan)

class BrowserCommand(BaseModel):
    """Model for browser commands."""
    code: str
    session_id: str = "default"
    
class BrowserResponse(BaseModel):
    """Model for browser command responses."""
    success: bool
    output: str
    error: Optional[str] = None
    screenshot: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint for health check."""
    global playwright_instance
    if playwright_instance is None:
        return {"status": "Browser service is running but Playwright is not initialized"}
    return {"status": "Browser service is running"}

@app.post("/browser/init", response_model=BrowserResponse)
async def init_browser(session_id: str = "default", connect_to_cdp: Optional[str] = None, browser_type: str = "default", anchor_api_key: Optional[str] = None):
    """Initialize a new browser instance."""
    success, message = await run_in_executor(init_browser_sync, session_id, connect_to_cdp, browser_type, anchor_api_key)
    
    return BrowserResponse(
        success=success,
        output=message
    )

@app.post("/browser/execute", response_model=BrowserResponse)
async def execute_command(command: BrowserCommand):
    """Execute a Playwright command in the browser."""
    success, output, screenshot = await run_in_executor(
        execute_code_sync, command.session_id, command.code
    )
    
    return BrowserResponse(
        success=success,
        output=output,
        screenshot=screenshot
    )

@app.get("/browser/status", response_model=BrowserResponse)
async def get_status(session_id: str = "default"):
    """Get the status of the browser."""
    success, message = await run_in_executor(get_status_sync, session_id)
    
    return BrowserResponse(
        success=success,
        output=message
    )

@app.post("/browser/close", response_model=BrowserResponse)
async def close_browser(session_id: str = "default"):
    """Close a browser instance."""
    success, message = await run_in_executor(close_browser_sync, session_id)
    
    return BrowserResponse(
        success=success,
        output=message
    )

if __name__ == "__main__":
    uvicorn.run("browser_service:app", host="0.0.0.0", port=8000) 