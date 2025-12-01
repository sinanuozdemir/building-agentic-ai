import os
from dotenv import dotenv_values
import streamlit as st
import builtins
import contextlib
import io
from typing import Any, Dict, Optional
import sys

# Add the current directory to the path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph
import time
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from create_codeact import create_codeact, CodeActState, CodeMessage, create_default_prompt
from state_manager import StateManager
from langgraph.checkpoint.memory import MemorySaver
from browser_client import BrowserServiceClient

# Import Playwright for type checking only
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    st.error("Playwright not installed. Install with: pip install playwright")
    st.stop()

# ONLY load from .env file, ignore system environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
env_vars = dotenv_values(env_path)
OPENAI_API_KEY = env_vars.get("OPENAI_API_KEY")
ANCHOR_API_KEY = env_vars.get("ANCHOR_API_KEY")

if not OPENAI_API_KEY:
    st.error("No OpenAI API key found in .env file")
    st.stop()

# Initialize state manager and memory saver
state_manager = StateManager()
memory_saver = MemorySaver()

# Create browser service client
browser_client = BrowserServiceClient("http://localhost:8000")

# Check if browser service is running
if not browser_client.health_check():
    st.error("Browser service is not running. Please start it with ./run_service.sh")
    st.stop()

# Page config
st.set_page_config(
    page_title="CodeAct Agent Chat",
    page_icon="🤖",
    layout="wide"
)

class CodeBrowserAgent:
    """A class that combines CodeAct agent with browser service for automation."""
    
    def __init__(self, api_key: str, browser_type: str = "anchor", connect_to_cdp: str = None, anchor_api_key: str = None):
        """Initialize the CodeBrowserAgent with the specified API key.
        
        Args:
            api_key: OpenAI API key for the language model
            browser_type: Type of browser to use ("default", "existing", "anchor")
            connect_to_cdp: Optional Chrome DevTools Protocol endpoint to connect to an existing browser
            anchor_api_key: API key for Anchor Browser (required if browser_type is "anchor")
        """
        self.api_key = api_key
        self.context = {}  # Persistent context (non-browser objects)
        self.model_messages = []  # Persistent model message history
        self.session_id = st.session_state.get("thread_id", str(time.time()))
        self.connect_to_cdp = connect_to_cdp
        self.browser_type = browser_type
        self.anchor_api_key = anchor_api_key
        
        # Initialize browser via service
        success, message = browser_client.init_browser(
            self.session_id, 
            connect_to_cdp, 
            browser_type, 
            anchor_api_key
        )
        
        if success:
            print(f"Browser initialized: {message}", file=sys.stderr)
            self.context['browser_ready'] = True
            
            # Store Anchor Browser live view URL if available
            if "Live view:" in message:
                live_view_url = message.split("Live view:")[1].strip()
                self.context['live_view_url'] = live_view_url
                st.session_state.live_view_url = live_view_url
            
            # If there's a saved URL in session state, navigate to it
            if browser_type != "anchor" and not connect_to_cdp and hasattr(st.session_state, "browser_context") and "last_url" in st.session_state.browser_context:
                last_url = st.session_state.browser_context.get("last_url")
                if last_url and last_url != "about:blank":
                    try:
                        print(f"Navigating to previously saved URL: {last_url}", file=sys.stderr)
                        code = f"page.goto('{last_url}')\noutput = 'Navigated to saved URL'"
                        browser_client.execute_code(code, self.session_id)
                    except Exception as e:
                        print(f"Error navigating to saved URL: {str(e)}", file=sys.stderr)
        else:
            print(f"Failed to initialize browser: {message}", file=sys.stderr)
            self.context['browser_ready'] = False
        
        self.agent = self._initialize_agent()
        
    def _initialize_agent(self) -> Optional[StateGraph]:
        """Initialize the CodeAct agent with the stored API key."""
        try:
            # Use Inception Labs API key from environment variable
            inception_api_key = os.getenv("INCEPTION_LABS_API_KEY")
            if not inception_api_key:
                raise ValueError("INCEPTION_LABS_API_KEY environment variable is required. Set it in your .env file.")
            model = ChatOpenAI(
                temperature=0,
                openai_api_key=inception_api_key,
                base_url="https://api.inceptionlabs.ai/v1",
                model="mercury-coder-small"
            )
            # model = ChatOpenAI(
            #     temperature=0,
            #     # openai_api_key=self.api_key,
            #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            #     base_url="https://openrouter.ai/api/v1",
            #     # model="meta-llama/llama-4-maverick:free"
            #     model="qwen/qwen3-30b-a3b:free"
            # )
            model = ChatOpenAI(
                temperature=0,
                openai_api_key=self.api_key,
                model="gpt-4.1"
            )

            
            # Create a custom prompt that informs the agent about the browser
            connection_type = "connected to an existing browser" if self.connect_to_cdp else "initialized for you"
            custom_prompt = f"""You are a browser automation assistant powered by a remote browser service.

DO NOT use the standard Playwright initialization. Instead, use page and browser objects directly.

A browser instance has been {connection_type} and is available on a remote server.
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

NEVER use this pattern (this will not work):
```python
# DO NOT DO THIS:
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    # ...
```"""
     
            code_act = create_codeact(
                model=model,
                tools=[],
                eval_fn=self.eval_code,
                prompt=create_default_prompt([], custom_prompt)
            )
            
            # Ensure thread_id exists in session state
            if "thread_id" not in st.session_state:
                st.session_state.thread_id = str(time.time())
                
            # Compile without config, we'll pass the thread_id when invoking
            return code_act.compile(checkpointer=memory_saver)
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            return None
    
    def eval_code(self, code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Execute code using the browser service."""
        # Merge the persistent context into locals
        for key, value in self.context.items():
            _locals[key] = value
            
        original_keys = set(_locals.keys())
        
        # Check browser status before executing
        success, status = browser_client.get_browser_status(self.session_id)
        if not success:
            print(f"Browser status check failed: {status}", file=sys.stderr)
            return f"⚠️ Error: Browser is not available. {status}", {}
        
        print(f"Browser status: {status}", file=sys.stderr)
        
        # Execute code via browser service
        success, output, screenshot_path = browser_client.execute_code(code, self.session_id)
        
        if success:
            result = output
            if screenshot_path:
                result += f"\n[Screenshot saved to {screenshot_path}]"
        else:
            result = f"⚠️ Error during execution: {output}"
            
        # The browser service doesn't return new variables, so we only return what we know
        new_vars = {
            "browser_ready": True,
            "browser_service_connected": True
        }
        
        # Get current URL to save in context
        try:
            status_success, status_msg = browser_client.get_browser_status(self.session_id)
            if "Current page:" in status_msg:
                # Extract URL from status
                url_start = status_msg.rfind("(") + 1
                url_end = status_msg.rfind(")")
                if url_start > 0 and url_end > url_start:
                    new_vars["last_url"] = status_msg[url_start:url_end]
        except Exception as e:
            print(f"Error getting URL: {str(e)}", file=sys.stderr)
        
        # Update the persistent context with new variables
        self.context.update(new_vars)
        
        return result, new_vars
    
    def clean_context(self, context):
        """Clean the context to prevent serialization issues."""
        # Create a clean copy without non-serializable objects
        clean_context = {}
        
        # Skip browser-related objects that may cause recursion
        skip_keys = ["browser", "browser_instance", "page", "playwright_instance", "browser_page"]
        
        for key, value in context.items():
            # Skip browser objects
            if key in skip_keys:
                continue
            
            # Include serializable values
            if isinstance(value, (str, int, float, bool, type(None))):
                clean_context[key] = value
            elif isinstance(value, dict):
                # For nested dictionaries, clean recursively
                clean_dict = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_dict[k] = v
                clean_context[key] = clean_dict
        
        # Always include browser_ready flag
        if "browser_ready" in context:
            clean_context["browser_ready"] = context["browser_ready"]
        
        return clean_context

    def process_message(self, message: str) -> str:
        """Process a user message using the CodeAct agent."""
        if not self.agent:
            return "Agent not initialized. Please check your API key."
        
        # Check if browser service is healthy
        if not browser_client.health_check():
            return "Browser service is not available. Please restart it with ./run_service.sh"
            
        # Add user message to the model's message history if it's not already empty
        if not self.model_messages:
            # Initialize with a system message if this is the first interaction
            prompt = create_default_prompt([])
            self.model_messages.append(SystemMessage(content=prompt))
            
        # Add the user's message
        self.model_messages.append(HumanMessage(content=message))
        
        # Clean the context to prevent serialization issues
        clean_context = self.clean_context(self.context)
        
        # Create initial state with persisted context and message history
        initial_state = CodeActState(
            messages=self.model_messages.copy(),  # Use the persistent message history
            last_user_input=message,
            context=clean_context  # Use the cleaned context
        )
        
        # Process the message through the agent - use invoke instead of stream_events
        try:
            # Ensure thread_id exists in session state
            if "thread_id" not in st.session_state:
                st.session_state.thread_id = str(time.time())
                
            # Create config with thread_id for the checkpointer and increased recursion limit
            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                },
                "recursion_limit": 100  # Increase recursion limit to allow more iterations
            }
            
            # Invoke agent with the config
            print(f"\n=== INVOKING AGENT ===\n", file=sys.stderr)
            result = self.agent.invoke(initial_state, config=config)
            # Pretty print the result for better readability
            import json
            
            # Print the agent result with simple delimiters
            print("\n----- AGENT RESULT -----", file=sys.stderr)
            if isinstance(result, dict) and "messages" in result:
                print(f"Result type: dict with {len(result['messages'])} messages", file=sys.stderr)
                for i, msg in enumerate(result["messages"]):
                    msg_type = type(msg).__name__
                    content = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content
                    print(f"Message {i+1}: {msg_type} - {content}", file=sys.stderr)
            else:
                print(f"Result type: {type(result)}", file=sys.stderr)
                print(str(result), file=sys.stderr)
            print("--------------------------\n", file=sys.stderr)
            
            # Get output messages from the result
            output_messages = []
            # Process messages - they might be in different formats
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                
                # Update our model messages with the result
                self.model_messages = messages.copy()
                
                # Find both code execution messages and AI response messages
                code_execution_msg = None
                latest_ai_message = None
                
                # Extract both code execution output and AI response
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        if "Code execution output:" in msg.content and not code_execution_msg:
                            code_execution_msg = msg
                        elif not latest_ai_message:
                            latest_ai_message = msg
                            
                        # Stop when we have both types of messages
                        if code_execution_msg and latest_ai_message:
                            break
                
                # Include code execution output if available
                if code_execution_msg:
                    # Extract just the output part, without the prompt at the end
                    content = code_execution_msg.content
                    if "is there anything else you need to do" in content:
                        output_part = content.split("is there anything else you need to do")[0]
                    else:
                        output_part = content
                    output_messages.append(output_part)
                
                # Add the AI response
                if latest_ai_message:
                    output_messages.append(latest_ai_message.content)
                else:
                    # Fallback if no AI message found
                    output_messages.append("I processed your request but didn't get a clear response.")
                
                # Update our persistent context with any changes from the agent
                if "context" in result:
                    # Clean the returned context before updating our persistent context
                    result_context = self.clean_context(result["context"])
                    self.context.update(result_context)
            else:
                # Handle unexpected result format
                output_messages.append("The agent returned an unexpected result format.")
                print(f"Unexpected result format: {type(result)}", file=sys.stderr)
            
        except Exception as e:
            import traceback
            print(f"Error running agent: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
            return f"Error: {str(e)}"
        
        # Return all outputs joined together, filtering out empty messages
        return "\n".join([msg for msg in output_messages if msg.strip()])
    
    def get_browser_status(self) -> str:
        """Get the status of the current browser session."""
        success, status = browser_client.get_browser_status(self.session_id)
        if success:
            return status
        else:
            return f"Error getting browser status: {status}"
            
    def __del__(self):
        """Cleanup resources when the agent is deleted."""
        try:
            # Close the browser via service
            print(f"Closing browser for session {self.session_id}...", file=sys.stderr)
            browser_client.close_browser(self.session_id)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}", file=sys.stderr)

# Initialize session state first
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session" not in st.session_state:
    st.session_state.current_session = None

# Optional reset button for the chat
st.sidebar.subheader("Chat Controls")
if st.sidebar.button("🔄 New Conversation"):
    # Clean up browser instance if it exists
    if "agent" in st.session_state and hasattr(st.session_state.agent, "context"):
        if "browser_instance" in st.session_state.agent.context:
            try:
                print("Closing existing browser...", file=sys.stderr)
                st.session_state.agent.context["browser_instance"].close()
            except Exception as e:
                print(f"Error closing browser: {str(e)}", file=sys.stderr)
                
        if "playwright_instance" in st.session_state.agent.context:
            try:
                print("Stopping Playwright...", file=sys.stderr)
                st.session_state.agent.context["playwright_instance"].stop()
            except Exception as e:
                print(f"Error stopping Playwright: {str(e)}", file=sys.stderr)
                
    # Clear session state while preserving browser preferences
    browser_type = st.session_state.get("browser_type", "anchor")
    cdp_endpoint = st.session_state.get("cdp_endpoint", "http://localhost:9222")
    anchor_api_key = st.session_state.get("anchor_api_key", ANCHOR_API_KEY)
    api_key = st.session_state.api_key
    
    # Reset session state
    st.session_state.messages = []
    st.session_state.pop("agent", None)
    st.session_state.thread_id = str(time.time())
    st.session_state.current_session = None
    st.session_state.pop("live_view_url", None)
    
    # Restore browser preferences
    st.session_state.browser_type = browser_type
    st.session_state.cdp_endpoint = cdp_endpoint
    st.session_state.anchor_api_key = anchor_api_key
    st.session_state.api_key = api_key
    
    # Initialize a new agent with the existing preferences
    if api_key:
        connect_to_endpoint = None
        anchor_api_key_param = None
        
        if browser_type == "existing":
            connect_to_endpoint = cdp_endpoint
        elif browser_type == "anchor":
            anchor_api_key_param = anchor_api_key
            
        st.session_state.agent = CodeBrowserAgent(
            api_key, 
            browser_type,
            connect_to_endpoint, 
            anchor_api_key_param
        )
    
    st.rerun()

# Display thread ID
st.sidebar.text(f"Current Session: {st.session_state.get('thread_id', 'New')}")

# API Key management
api_key = OPENAI_API_KEY
if "api_key" not in st.session_state:
    st.session_state.api_key = api_key
if "anchor_api_key" not in st.session_state:
    st.session_state.anchor_api_key = ANCHOR_API_KEY

# Initialize agent if needed
if "agent" not in st.session_state and st.session_state.api_key:
    st.session_state.agent = CodeBrowserAgent(st.session_state.api_key)
    
# Session management
st.sidebar.subheader("Session Management")
saved_states = state_manager.list_saved_states()
if saved_states:
    session_to_load = st.sidebar.selectbox("Load previous session", [""] + saved_states)
    if session_to_load and st.sidebar.button("Load Session"):
        if "agent" in st.session_state and hasattr(st.session_state.agent, "context"):
            # Clean up existing browser instance before loading a new session
            if "browser_instance" in st.session_state.agent.context:
                try:
                    print("Closing existing browser before loading session...", file=sys.stderr)
                    st.session_state.agent.context["browser_instance"].close()
                except Exception as e:
                    print(f"Error closing browser: {str(e)}", file=sys.stderr)
                    
            if "playwright_instance" in st.session_state.agent.context:
                try:
                    print("Stopping Playwright before loading session...", file=sys.stderr)
                    st.session_state.agent.context["playwright_instance"].stop()
                except Exception as e:
                    print(f"Error stopping Playwright: {str(e)}", file=sys.stderr)
                    
        # Load the selected session
        if state_manager.load_state(session_to_load):
            st.session_state.thread_id = session_to_load
            st.session_state.current_session = session_to_load
            
            # Get browser connection preferences (from current session state)
            use_existing_browser = st.session_state.get("use_existing_browser", False)
            cdp_endpoint = st.session_state.get("cdp_endpoint", "http://localhost:9222") if use_existing_browser else None
            
            # Reinitialize the agent with the loaded API key and browser preferences
            st.session_state.agent = CodeBrowserAgent(st.session_state.api_key, cdp_endpoint)
            
            # Restore browser context values (non-browser objects)
            if hasattr(st.session_state, "browser_context"):
                for key, value in st.session_state.browser_context.items():
                    st.session_state.agent.context[key] = value
                    
            st.success(f"Loaded session: {session_to_load}")
            st.rerun()
        else:
            st.error("Failed to load session")

# Save current session
session_name = st.sidebar.text_input("Session name", st.session_state.get("thread_id", ""))
if session_name and st.sidebar.button("Save Session"):
    st.session_state.thread_id = session_name
    st.session_state.current_session = session_name
    if state_manager.save_state(session_name):
        st.sidebar.success(f"Saved as: {session_name}")
    else:
        st.sidebar.error("Failed to save session")

# Delete session
if saved_states:
    session_to_delete = st.sidebar.selectbox("Delete session", [""] + saved_states, key="delete_session")
    if session_to_delete and st.sidebar.button("Delete Session"):
        if state_manager.delete_state(session_to_delete):
            st.sidebar.success(f"Deleted session: {session_to_delete}")
            st.rerun()
        else:
            st.sidebar.error("Failed to delete session")

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .stMarkdown {
        font-size: 16px;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stCodeBlock {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stAlert {
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for API key input and state management
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Key input
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,  # Use the key from .env by default
        help="Enter your OpenAI API key. Default is loaded from .env file."
    )
    
    # Browser connection options
    st.subheader("Browser Options")
    
    browser_type = st.radio(
        "Browser Type",
        ["Anchor Browser", "Local Browser", "Connect to Existing"],
        index=0,  # Default to Anchor Browser
        help="Choose which browser type to use"
    )
    
    # Store the selected browser type
    st.session_state.browser_type = {
        "Anchor Browser": "anchor",
        "Local Browser": "default", 
        "Connect to Existing": "existing"
    }.get(browser_type)
    
    # Anchor Browser settings
    if browser_type == "Anchor Browser":
        anchor_api_key = st.text_input(
            "Anchor Browser API Key",
            type="password",
            value=st.session_state.get("anchor_api_key", ""),
            help="Enter your Anchor Browser API key"
        )
        st.session_state.anchor_api_key = anchor_api_key
        
        st.info("Anchor Browser provides a cloud-based browser that can be accessed remotely. You'll see a live view of the browser below the chat.")

    # Connect to existing browser option
    connect_to_cdp = None
    if browser_type == "Connect to Existing":
        connect_to_cdp = st.text_input(
            "Chrome DevTools Protocol endpoint",
            value=st.session_state.get("cdp_endpoint", "http://localhost:9222"),
            help="The CDP endpoint of your Chrome browser. Run Chrome with --remote-debugging-port=9222"
        )
        st.markdown("""
        **To use an existing browser:**
        
        For Chrome:
        ```
        /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
        ```
        
        For Arc:
        ```
        open -a "Arc" --args --remote-debugging-port=9222
        ```
        
        For Brave:
        ```
        /Applications/Brave\ Browser.app/Contents/MacOS/Brave\ Browser --remote-debugging-port=9222
        ```
        
        Then use the default endpoint: `http://localhost:9222`
        
        **Note**: When using an existing browser, make sure the tab you want to automate is already open.
        """)
    
    # Store values in session state
    if browser_type == "Connect to Existing" and connect_to_cdp:
        st.session_state.cdp_endpoint = connect_to_cdp
    
    if api_key_input:
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            with st.spinner("Initializing agent..."):
                # Get parameters based on browser type
                browser_type_param = st.session_state.get("browser_type", "anchor")
                connect_to_endpoint = None
                anchor_api_key_param = None
                
                if browser_type_param == "existing":
                    connect_to_endpoint = st.session_state.get("cdp_endpoint")
                elif browser_type_param == "anchor":
                    anchor_api_key_param = st.session_state.get("anchor_api_key")
                
                st.session_state.agent = CodeBrowserAgent(
                    api_key_input, 
                    browser_type_param,
                    connect_to_endpoint, 
                    anchor_api_key_param
                )
                
                if st.session_state.agent:
                    st.success("Agent initialized successfully!")
    
    # State management
    st.subheader("💾 Chat States")
    
    # Save current state
    session_name = st.text_input("Save as", placeholder="Enter a name for this chat")
    if st.button("💾 Save Chat"):
        if session_name:
            if state_manager.save_state(session_name):
                st.session_state.current_session = session_name
                st.success(f"Chat saved as '{session_name}'")
            else:
                st.error("Failed to save chat")
        else:
            st.warning("Please enter a name for this chat")
    
    # Load saved states
    saved_states = state_manager.list_saved_states()
    if saved_states:
        st.subheader("Saved Chats")
        selected_state = st.selectbox("Select a chat", saved_states)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Load"):
                if state_manager.load_state(selected_state):
                    st.session_state.current_session = selected_state
                    # Reinitialize agent with the loaded state
                    if "api_key" in st.session_state:
                        st.session_state.agent = CodeBrowserAgent(st.session_state.api_key)
                        # Apply any saved browser context
                        if "browser_context" in st.session_state:
                            st.session_state.agent.context.update(st.session_state.browser_context)
                    st.rerun()  # Rerun to refresh UI with loaded state
        with col2:
            if st.button("🗑️ Delete"):
                if state_manager.delete_state(selected_state):
                    if st.session_state.current_session == selected_state:
                        st.session_state.current_session = None
                    st.success(f"Deleted '{selected_state}'")
                    time.sleep(1)
                    st.rerun()  # Rerun to refresh the list

# Display chat title
if st.session_state.current_session:
    st.title(f"🤖 CodeAct Browser - {st.session_state.current_session}")
else:
    st.title("🤖 CodeAct Browser")

# Display current browser status
if st.session_state.agent:
    status = st.session_state.agent.get_browser_status()
    st.info(f"Browser Status: {status}")

# Display Anchor Browser live view if available
if st.session_state.get("browser_type") == "anchor" and st.session_state.get("live_view_url"):
    st.subheader("🔍 Browser Live View")
    live_view_url = st.session_state.get("live_view_url")
    st.markdown(
        f"""
        <iframe src="{live_view_url}" 
        sandbox="allow-same-origin allow-scripts" 
        allow="clipboard-read; clipboard-write" 
        style="border: 0px; display: block; width: 100%; height: 500px;">
        </iframe>
        """, 
        unsafe_allow_html=True
    )

# Create a message container to display messages
message_container = st.container()

# Use a separate section for the input
input_section = st.container()

# Display chat messages in the message container
with message_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""<div class="user-message">{message["content"]}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="assistant-message">{message["content"]}</div>""", unsafe_allow_html=True)

# Move the chat input to the input section
with input_section:
    user_input = st.chat_input("What would you like to do?")

if user_input and st.session_state.agent:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get agent response
    with st.spinner("Thinking..."):
        response = st.session_state.agent.process_message(user_input)
    
    # Add agent response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update UI with new messages
    st.rerun()
    
elif user_input:
    st.error("Agent not initialized. Please check your API key.") 