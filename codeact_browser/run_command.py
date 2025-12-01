#!/usr/bin/env python
import os
import sys
import argparse
import time
from dotenv import dotenv_values
import uuid
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from create_codeact import create_codeact, create_default_prompt
from browser_client import BrowserServiceClient

# ONLY load from .env file, ignore system environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
env_vars = dotenv_values(env_path)
OPENAI_API_KEY = env_vars.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: No OpenAI API key found in .env file", file=sys.stderr)
    sys.exit(1)

def eval_code(code, _locals, session_id, browser_client):
    """Execute code using the browser service."""
    # Check browser status before executing
    success, status = browser_client.get_browser_status(session_id)
    if not success:
        print(f"Browser status check failed: {status}", file=sys.stderr)
        return f"Error: Browser is not available. {status}", {}
    
    # Execute code via browser service
    success, output, screenshot_path = browser_client.execute_code(code, session_id)
    
    if success:
        if screenshot_path:
            print(f"Screenshot saved to: {screenshot_path}")
        return output, _locals
    else:
        return f"Error executing code: {output}", _locals

def run_single_command(command, headless=False, browser_url=None, timeout=60, model_name="gpt-4o"):
    """Run a single command using CodeAct and browser service.
    
    Args:
        command: The natural language command to execute
        browser_url: URL to connect to existing browser (None to launch new)
        timeout: Maximum seconds to wait for completion
        model_name: Name of the OpenAI model to use
    """
    # user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # Check if browser service is running
    browser_client = BrowserServiceClient("http://localhost:8000")
    if not browser_client.health_check():
        print("Error: Browser service is not running. Please start it with ./run_service.sh", file=sys.stderr)
        sys.exit(1)
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    print(f"Starting browser session with ID: {session_id}")
    # print(f"Using user agent: {user_agent}")
    
    # Initialize model
    model = ChatOpenAI(
        temperature=1,
        openai_api_key=OPENAI_API_KEY,
        model=model_name
    )
    
    # Create custom prompt
    custom_prompt = """You are a browser automation assistant powered by a remote browser service. You are going to be given a \
SINGLE task to complete, your job is to complete the task using the browser service. It is critical that you only complete the \
task once and do not do anything else. Summarize and output the final result of the task in the last message.

DO NOT use the standard Playwright initialization. Instead, use page and browser objects directly.

A browser instance has been initialized for you and is available on a remote server.
Your code will be executed in a context where these objects are available:
- page: The current browser page

IMPORTANT: DO NOT create new browser instances with sync_playwright() or other methods.
Simply use the 'page' object directly to navigate and interact with web pages.

Example of correct usage:
```python
# Navigate to a website using the existing page
page.goto('https://example.com')
print(f"Page title: {{page.title()}}")

# Take a screenshot
take_screenshot = True  # Set this to True to take a screenshot
print("Screenshot requested")

# Click a button
page.click('button.submit')

# Fill a form field
page.fill('input[name="search"]', 'query')

# Get text from the page
text = page.inner_text('h1')
print(f"Heading: {{text}}")
```

This is a one-time execution. Complete the entire requested task in a single code submission.
Be thorough and output any relevant information the user would want to see.
"""

    # Create a lambda function that captures browser_client and session_id
    eval_function = lambda code, _locals: eval_code(code, _locals, session_id, browser_client)
    
    # Create the CodeAct agent
    memory_saver = MemorySaver()
    code_act = create_codeact(
        model=model,
        tools=[],
        eval_fn=eval_function,
        prompt=create_default_prompt([], custom_prompt)
    )
    agent = code_act.compile(checkpointer=memory_saver)
    
    # Initialize browser
    browser_code = "page.goto('about:blank')\noutput = 'Browser initialized successfully'"
    if headless:
        print("Using headless browser")
    
    # Initialize browser
    success, message = browser_client.init_browser(session_id)
    if not success:
        print(f"Failed to initialize browser: {message}", file=sys.stderr)
        sys.exit(1)
    
    # Execute the initial navigation to make sure browser is ready
    browser_client.execute_code(browser_code, session_id)
    
    print(f"Executing command: {command}")
    print("---")
    
    # Create initial agent state with user command
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": session_id}}
    start_time = time.time()
    
    # Execute the command
    result = agent.invoke({"messages": [HumanMessage(content=command)]}, config)
    
    # Print all messages from the result
    if "messages" in result:
        for msg in result["messages"]:
            if hasattr(msg, "content"):
                print(msg.content)
    
    # Close the browser when done
    print("---")
    success, message = browser_client.close_browser(session_id)
    print(f"Browser session closed: {message}")
    
    # Report timing
    elapsed = time.time() - start_time
    print(f"Command completed in {elapsed:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Run a single command using CodeAct and browser service")
    parser.add_argument("command", help="The natural language command to execute", nargs="+")
    parser.add_argument("--timeout", type=int, default=60, help="Maximum seconds to wait for completion")
    parser.add_argument("--model", default="gpt-4o", help="Name of the OpenAI model to use")
    parser.add_argument("--user-agent", help="Custom user agent string to use for the browser")
    
    args = parser.parse_args()
    full_command = " ".join(args.command)
    
    run_single_command(
        command=full_command, 
        timeout=args.timeout,
        model_name=args.model,
    )

if __name__ == "__main__":
    main() 