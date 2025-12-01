import inspect
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union, Annotated, List, Dict
from dataclasses import dataclass
from enum import Enum, auto

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as create_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.channels import LastValue
from pydantic import BaseModel, Field

import re
import sys

BACKTICK_PATTERN = r"(?:^|\n)```(.*?)(?:```(?:\n|$))"


def extract_and_combine_codeblocks(text: str) -> str:
    """
    Extracts all codeblocks from a text string and combines them into a single code string.

    Args:
        text: A string containing zero or more codeblocks, where each codeblock is
            surrounded by triple backticks (```).

    Returns:
        A string containing the combined code from all codeblocks, with each codeblock
        separated by a newline.

    Example:
        text = '''Here's some code:

        ```python
        print('hello')
        ```
        And more:

        ```
        print('world')
        ```'''

        result = extract_and_combine_codeblocks(text)

        Result:

        print('hello')

        print('world')
    """
    # Find all code blocks in the text using regex
    # Pattern matches anything between triple backticks, with or without a language identifier
    code_blocks = re.findall(BACKTICK_PATTERN, text, re.DOTALL)

    if not code_blocks:
        return ""

    # Process each codeblock
    processed_blocks = []
    for block in code_blocks:
        # Strip leading and trailing whitespace
        block = block.strip()

        # If the first line looks like a language identifier, remove it
        lines = block.split("\n")
        if lines and (not lines[0].strip() or " " not in lines[0].strip()):
            # First line is empty or likely a language identifier (no spaces)
            block = "\n".join(lines[1:])

        processed_blocks.append(block)

    # Combine all codeblocks with newlines between them
    combined_code = "\n\n".join(processed_blocks)
    return combined_code


class CodeMessage(BaseMessage):
    """Message type for code execution."""
    type: str
    content: str


class CodeActState(BaseModel):
    """State for CodeAct agent."""
    messages: List[Any] = Field(default_factory=list)
    script: Optional[str] = None
    last_user_input: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    next_node: Optional[str] = None


StateSchema = TypeVar("StateSchema", bound=CodeActState)
StateSchemaType = Type[StateSchema]


def create_default_prompt(tools: list[StructuredTool], base_prompt: Optional[str] = None):
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You are a helpful coding assistant who can write python code to perform tasks on a given browser.
    
You will be given a task to perform. You can write code in sub-steps to complete the task. Each sub-step should be a small, focused code snippet (2-3 operations at a time).

On every message, you should output either

- reasoning + a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you \
want to extract from the code should be printed to the console. Code should be output in a fenced code block. \
ALL CODE should be wrapped between ```python and ```.

- text to be shown directly to the user, if you want to ask for more information or provide the final answer.

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in tools:
        prompt += f'''
def {tool.name}{str(inspect.signature(tool.func))}:
    """{tool.description}"""
    ...
'''

    prompt += """

Context / Help on how to use the browser:

Variables defined at the top level of previous code snippets can be referenced in your code.

You have access to a pre-initialized browser session with the following variables:
- playwright_instance: The Playwright instance that's already running
- browser_instance: The browser instance that's already running
- browser_page: The current browser page that's ready to use

IMPORTANT: USE THESE EXISTING OBJECTS instead of creating new browser instances. For example:

```python
# Use the existing browser page
browser_page.goto('https://example.com')
print(f"Page title: {browser_page.title()}")

# Take a screenshot
browser_page.screenshot(path='screenshot.png')
print("Screenshot taken")

# Click a button
browser_page.click('button.submit')

# Fill a form field
browser_page.fill('input[name="search"]', 'query')

# Get text from the page
text = browser_page.inner_text('h1')
print(f"Heading: {text}")
```

DO NOT create new browser instances with code like this unless explicitly asked:
```python
# DO NOT USE THIS PATTERN - browser is already initialized
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    # ...
```

BROWSER AUTOMATION BEST PRACTICES - FOLLOW THESE TO AVOID TIMEOUTS AND HANGS:
1. Always write small, focused code snippets (2-3 operations at a time)
2. Always verify elements exist before interacting with them:
   ```python
   # Check if element exists first
   if browser_page.query_selector('input[name="q"]'):
       browser_page.fill('input[name="q"]', 'search term')
   else:
       print("Search input not found")
   ```
3. Use better selectors:
   - Avoid XPath when possible
   - Use specific selectors like IDs (#id) and classes (.class) when available
   - After `goto()`, always print page title to confirm navigation worked
   - Use browser_page.wait_for_selector('selector') with caution
4. Debug by inspecting the page:
   ```python
   # List all input elements to find the right selector
   inputs = browser_page.query_selector_all('input')
   for i, input_el in enumerate(inputs):
       name = input_el.get_attribute('name')
       id = input_el.get_attribute('id')
       print(f"Input {i}: name='{name}', id='{id}'")
   ```
5. After page navigation, always allow content to load with a short print statement:
   ```python
   browser_page.goto('https://example.com')
   print(f"Page loaded: {browser_page.title()}")
   ```

EXAMPLES OF PRINTING PAGE ELEMENTS FOR DEBUGGING:

Example 1: Print all form elements
```python
# User request: "I'm trying to sign up for an online course at MIT OpenCourseWare, but I can't figure out which form to use"
# I'll analyze all available forms on the page to help identify the registration form
# Understanding form attributes like action URLs and methods will help determine the right one
# This approach helps debug complex pages with multiple forms when the user is confused

# Find all form elements
form_elements = browser_page.query_selector_all('form')
print(f"Found {len(form_elements)} forms on the page")

for form_index, form_element in enumerate(form_elements):
    form_action = form_element.get_attribute('action') or 'No action'
    form_method = form_element.get_attribute('method') or 'No method'
    form_id = form_element.get_attribute('id') or 'No ID'
    print(f"Form {form_index}: action='{form_action}', method='{form_method}', id='{form_id}'")
    
    # Find all inputs within this form
    form_input_elements = form_element.query_selector_all('input')
    print(f"  Form {form_index} has {len(form_input_elements)} input elements")
    for input_index, input_element in enumerate(form_input_elements):
        input_type = input_element.get_attribute('type') or 'text'
        input_name = input_element.get_attribute('name') or 'No name'
        input_id = input_element.get_attribute('id') or 'No ID'
        print(f"  Input {input_index}: type='{input_type}', name='{input_name}', id='{input_id}'")
```

Example 2: Find clickable elements
```python
# User request: "I need to analyze the checkout process for my Shopify store to ensure all buttons are correctly labeled for accessibility"
# I'll scan the page to find all clickable elements to identify buttons that need better labeling
# This helps with accessibility audits by locating buttons that might not have descriptive text
# Finding both buttons and links gives a complete picture of all interactive elements on the checkout page

# Find all buttons and links
button_elements = browser_page.query_selector_all('button, [role="button"], input[type="submit"]')
link_elements = browser_page.query_selector_all('a[href]')

print(f"Found {len(button_elements)} buttons and {len(link_elements)} links")

for button_index, button_element in enumerate(button_elements):
    button_text = button_element.inner_text().strip() or 'No text'
    button_id = button_element.get_attribute('id') or 'No ID'
    button_classes = button_element.get_attribute('class') or 'No classes'
    print(f"Button {button_index}: text='{button_text}', id='{button_id}', classes='{button_classes}'")

for link_index, link_element in enumerate(link_elements):
    link_text = link_element.inner_text().strip() or 'No text'
    link_href = link_element.get_attribute('href') or 'No href'
    print(f"Link {link_index}: text='{link_text}', href='{link_href}'")
```

Example 3: Check element visibility and properties
```python
# User request: "I've redesigned my portfolio website to be mobile-responsive, but some clients say the contact form is hard to find on smartphones"
# I'll check if the search box is properly visible and accessible on different screen sizes
# Testing element properties like visibility and dimensions helps diagnose responsive design issues
# This approach can verify if elements are correctly displayed before users attempt to interact with them

# Find a specific element and check its properties
search_box_element = browser_page.query_selector('.search-box')
if search_box_element:
    element_is_visible = search_box_element.is_visible()
    element_is_enabled = search_box_element.is_enabled()
    element_is_editable = search_box_element.is_editable()
    element_bounding_box = search_box_element.bounding_box()
    element_text = search_box_element.inner_text().strip()
    
    print(f"Element properties:")
    print(f"  Visible: {element_is_visible}")
    print(f"  Enabled: {element_is_enabled}")
    print(f"  Editable: {element_is_editable}")
    print(f"  Text: '{element_text}'")
    print(f"  Position: x={element_bounding_box['x']}, y={element_bounding_box['y']}")
    print(f"  Size: width={element_bounding_box['width']}, height={element_bounding_box['height']}")
else:
    print("Element not found")
```

Example 4: Get and print page structure
```python
# User request: "I'm building a web scraper for price monitoring of electronics, but the site has a complex layout"
# I need to understand the page's DOM hierarchy to create reliable selectors for data extraction
# Mapping the element tree helps identify the best paths to access product information and prices
# This approach is essential when working with dynamic websites that change their structure frequently

# Print the basic structure of a page
def print_element_tree(page_element, element_selector, tree_depth=0):
    # Get tag name and basic attributes
    element_tag_name = page_element.evaluate('el => el.tagName.toLowerCase()')
    element_id = page_element.get_attribute('id')
    element_class = page_element.get_attribute('class')
    
    # Build a string representation
    indent_spaces = "  " * tree_depth
    element_description = f"{indent_spaces}<{element_tag_name}"
    if element_id:
        element_description += f" id='{element_id}'"
    if element_class:
        element_description += f" class='{element_class}'"
    element_description += ">"
    
    print(element_description)
    
    # Process only direct children for brevity
    if tree_depth < 2:  # Limit depth to avoid too much output
        child_elements = page_element.query_selector_all(':scope > *')
        for child_element in child_elements:
            print_element_tree(child_element, element_selector, tree_depth + 1)

# Get the body element and print its structure
body_element = browser_page.query_selector('body')
if body_element:
    print("Page structure:")
    print_element_tree(body_element, 'body')
else:
    print("Body element not found")
```

EXAMPLES FOR GOOGLE SEARCH:

Example 5: Direct Google search with URL
```python
# User request: "I need to find information about Cafe Revile near my home at 123 Main Street. Can you look up their website and coffee prices?"
# This direct search approach is efficient when we need specific information quickly
# We'll search for the cafe name with the address and look for their website or menu information
# Using a direct URL search saves time compared to navigating to Google's homepage first

# Perform a direct Google search via URL
search_query = "Cafe Revile 123 Main Street menu coffee prices"  # Specific search for the cafe
encoded_search_query = search_query.replace(' ', '+')
search_url = f"https://www.google.com/search?q={encoded_search_query}&udm=14"

# Navigate to the search URL
browser_page.goto(search_url)
print(f"Navigated to Google search for: '{search_query}'")
print(f"Page title: {browser_page.title()}")

# Check if search results are present
search_results = browser_page.query_selector_all("div.g")
print(f"Found {len(search_results)} search results")

# Extract the first few search results (if any)
for i, result in enumerate(search_results[:3]):  # Look at first 3 results only
    title_element = result.query_selector("h3")
    if title_element:
        title = title_element.inner_text()
        print(f"Result {i+1}: {title}")
```

Example 6: Search Google for information using the search box
```python
# User request: "I'm researching the impact of climate change on polar bear populations for my environmental science project"
# We'll search for recent information about polar bear populations and climate change impacts
# Using Google's search box gives us flexibility to search for specific scientific data
# This can help find academic studies, news articles, and conservation websites with the latest information

# Navigate to Google homepage
browser_page.goto("https://www.google.com")
print(f"Navigated to Google homepage: {browser_page.title()}")

# Check if search box exists - using multiple selector approaches for reliability
search_input = browser_page.query_selector('input[name="q"]') or browser_page.query_selector('#searchbox input') or browser_page.query_selector('input[type="text"]')
if search_input:
    # Type a search query
    search_query = "polar bear population decline arctic climate change research 2023"
    browser_page.fill('input[name="q"]', search_query)
    print(f"Entered search query: '{search_query}'")
    
    # Press Enter to search
    browser_page.press('input[name="q"]', 'Enter')
    
    # Wait for page to load and verify title
    print(f"Search results page title: {browser_page.title()}")
else:
    print("Search input not found. Investigating available elements...")
    
    # Debug: List all input elements to find the right selector
    input_elements = browser_page.query_selector_all('input')
    for input_index, input_element in enumerate(input_elements):
        input_name = input_element.get_attribute('name') or 'No name'
        input_id = input_element.get_attribute('id') or 'No ID'
        input_type = input_element.get_attribute('type') or 'No type'
        print(f"Input {input_index}: name='{input_name}', id='{input_id}', type='{input_type}'")
```

Example 7: Extract information from Google search results
```python
# User request: "My family is flying from Tokyo to Sydney tomorrow on flight JA7109, and I need to check if there are any delays"
# I'll search for real-time flight information and try to extract status details from Google's featured information
# Google often shows flight status information directly in search results via Knowledge Panels
# This approach can quickly provide flight status without having to navigate to an airline website

# Perform a Google search
browser_page.goto("https://www.google.com/search?q=flight+JA7109+Tokyo+to+Sydney+status&udm=14")
print(f"Navigated to search page: {browser_page.title()}")

# Extract the direct answer if available (Knowledge Graph)
knowledge_panel = browser_page.query_selector(".kp-header")
if knowledge_panel:
    answer_text = knowledge_panel.inner_text()
    print(f"Knowledge panel found: {answer_text}")

# Extract organic search results
search_results = browser_page.query_selector_all("div.g")
print(f"Found {len(search_results)} organic search results")

for result_index, result_element in enumerate(search_results[:3]):  # Look at first 3 results
    # Try to get the title and URL
    title_element = result_element.query_selector("h3")
    link_element = result_element.query_selector("a")
    
    result_title = title_element.inner_text() if title_element else "No title"
    result_url = link_element.get_attribute("href") if link_element else "No URL"
    
    # Try to get the snippet
    snippet_element = result_element.query_selector("div.VwiC3b")
    result_snippet = snippet_element.inner_text() if snippet_element else "No snippet"
    
    print(f"Result {result_index + 1}:")
    print(f"  Title: {result_title}")
    print(f"  URL: {result_url}")
    print(f"  Snippet: {result_snippet[:100]}...")  # Truncate long snippets
```

Example 8: Use Google search to find information you don't know
```python
# User request: "I'm planning my farm's planting schedule for next season in Nebraska. Can you help me find historical rainfall data for the last 5 years in Lincoln, Nebraska during spring months?"
# I need to find specific regional weather data that I don't already have
# Using Google search can help locate authoritative weather data sources like NOAA or local agricultural extensions
# The function approach allows us to parse different result types (featured snippets, knowledge panels, or regular results)

# Function to search Google and extract information
def google_search_for_information(search_query):
    # Prepare the search URL
    encoded_query = search_query.replace(' ', '+')
    search_url = f"https://www.google.com/search?q={encoded_query}&udm=14"
    
    # Navigate to search page
    browser_page.goto(search_url)
    print(f"Searching Google for: '{search_query}'")
    print(f"Page title: {browser_page.title()}")
    
    # Check for direct answers (featured snippets, knowledge panels)
    featured_snippet = browser_page.query_selector(".V3FYCf") or browser_page.query_selector(".IZ6rdc")
    knowledge_panel = browser_page.query_selector(".kp-header") or browser_page.query_selector(".ifM9O")
    
    if featured_snippet:
        snippet_text = featured_snippet.inner_text()
        print(f"Featured snippet: {snippet_text}")
        return f"According to Google's featured snippet: {snippet_text}"
        
    elif knowledge_panel:
        panel_text = knowledge_panel.inner_text()
        print(f"Knowledge panel: {panel_text}")
        return f"According to Google Knowledge Panel: {panel_text}"
    
    else:
        # Extract the first few organic results
        search_results = browser_page.query_selector_all("div.g")
        
        if search_results:
            print(f"Found {len(search_results)} search results")
            
            # Get the first result
            first_result = search_results[0]
            title_element = first_result.query_selector("h3")
            snippet_element = first_result.query_selector("div.VwiC3b") or first_result.query_selector("span.aCOpRe")
            
            title = title_element.inner_text() if title_element else "No title"
            snippet = snippet_element.inner_text() if snippet_element else "No snippet available"
            
            return f"Top search result: {title}\nSnippet: {snippet}"
        else:
            return "No relevant information found for this query."

# Example usage: Find information about a topic
unknown_topic = "historical rainfall data Lincoln Nebraska spring months 5 year average"
search_result = google_search_for_information(unknown_topic)
print(f"\nSearch result for '{unknown_topic}':")
print(search_result)
```

Reminder: use Python code snippets to call tools"""
    return prompt


@dataclass
class CodeMessage:
    """Message containing code or code output."""
    content: str
    type: str  # 'code' or 'output'

    def to_langchain(self):
        """Convert to a LangChain-compatible message format."""
        prefix = "```python\n" if self.type == "code" else "Output:\n"
        suffix = "\n```" if self.type == "code" else ""
        return HumanMessage(content=f"{prefix}{self.content}{suffix}")


def create_codeact(
    model: BaseChatModel,
    tools: Sequence[Union[StructuredTool, Callable]],
    eval_fn: Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]],
    *,
    prompt: Optional[str] = None,
    state_schema: StateSchemaType = CodeActState,
) -> StateGraph:
    """Create a CodeAct agent.

    Args:
        model: The language model to use for generating code
        tools: List of tools available to the agent. Can be passed as python functions or StructuredTool instances.
        eval_fn: Function that executes code in a sandbox. Takes code string and locals dict,
            returns a tuple of (stdout output, new variables dict)
        prompt: Optional custom system prompt. If None, uses default prompt.
            To customize default prompt you can use `create_default_prompt` helper:
            `create_default_prompt(tools, "You are a helpful assistant.")`
        state_schema: The state schema to use for the agent.

    Returns:
        A StateGraph implementing the CodeAct architecture
    """
    print("\n=== CREATING NEW CODEACT AGENT ===\n")
    
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]

    if prompt is None:
        prompt = create_default_prompt(tools)

    def clean_context_for_serialization(context):
        """Clean browser objects from context to prevent serialization issues.
        
        This prevents recursion errors when trying to serialize browser objects.
        """
        # Create a clean copy without non-serializable objects
        clean_context = {}
        
        # Copy only simple values, avoiding complex browser objects
        for key, value in context.items():
            # Skip browser objects which can't be serialized
            if key in ["browser", "page", "playwright_instance", "browser_instance", "browser_page"]:
                continue
            
            # Include simple types that can be serialized
            if isinstance(value, (str, int, float, bool, type(None))):
                clean_context[key] = value
            elif isinstance(value, dict):
                # For dictionaries, recursively clean them
                clean_context[key] = clean_context_for_serialization(value)
            elif isinstance(value, list):
                # For lists, only include items that are simple types
                clean_list = []
                for item in value:
                    if isinstance(item, (str, int, float, bool, type(None))):
                        clean_list.append(item)
                clean_context[key] = clean_list
        
        # Always include the browser_ready flag if it exists
        if "browser_ready" in context:
            clean_context["browser_ready"] = context["browser_ready"]
        
        return clean_context

    def call_model(state: CodeActState) -> dict:
        """Generate Python code using the language model."""
        messages = state.messages.copy() if state.messages else []
        
        # Prepend system message if it doesn't exist yet
        if not messages or not any(isinstance(msg, SystemMessage) for msg in messages):
            messages.insert(0, SystemMessage(content=prompt))

        # If we have a user input and it hasn't been processed yet, add it
        user_message_added = False
        if state.last_user_input:
            # Check if the last user input is already in the messages
            already_processed = False
            for msg in messages:
                if isinstance(msg, HumanMessage) and msg.content == state.last_user_input:
                    already_processed = True
                    break
                    
            if not already_processed:
                messages.append(HumanMessage(content=state.last_user_input))
                user_message_added = True
                
            # Clear the last user input after checking
            state.last_user_input = None
        
        # If timeout or element selection errors occurred, add guidance
        has_timeout_error = False
        for msg in messages:
            if isinstance(msg, AIMessage) and "Code execution output:" in msg.content:
                if "Timeout" in msg.content or "timeout" in msg.content or "waiting for" in msg.content:
                    has_timeout_error = True
                    break
        
        if has_timeout_error:
            # Add helpful guidance for avoiding timeouts
            guidance_msg = HumanMessage(
                content="""I notice you're encountering timeout errors. Try these techniques instead:
                
1. Check if elements exist before interacting with them:
```python
# First, verify element exists
if browser_page.query_selector('input[name="q"]'):
    browser_page.fill('input[name="q"]', 'search term')
else:
    # List all input elements to find the right selector
    inputs = browser_page.query_selector_all('input')
    for i, input_el in enumerate(inputs):
        name = input_el.get_attribute('name')
        id = input_el.get_attribute('id')
        print(f"Input {i}: name='{name}', id='{id}'")
```

2. Break down your actions into smaller steps and verify each step worked before continuing
3. After navigation, print the page title to confirm the page loaded correctly
"""
            )
            messages.append(guidance_msg)
        
        # Check if we have detected browser thread issues in previous steps
        browser_thread_issues = False
        browser_error_count = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and "Code execution output:" in msg.content:
                if "Cannot switch to a different thread" in msg.content or "greenlet.error" in msg.content:
                    browser_thread_issues = True
                    browser_error_count += 1
                    
        # If there are multiple thread issues, add a special message to guide the agent
        if browser_thread_issues and browser_error_count >= 2:
            # Add a special guidance message before calling the model
            guidance_msg = HumanMessage(
                content="""⚠️ WARNING: The browser automation is experiencing thread synchronization issues. 
                
Please use an alternative approach:
1. DO NOT use browser_page or any browser interactions directly
2. Use simpler Python code that doesn't interact with the browser
3. Or provide a text response explaining what you would do if the browser was working

The browser service needs to be restarted. Please provide a text response instead of code."""
            )
            messages.append(guidance_msg)
        
        # Log the number of messages being sent to the LLM
        print(f"\n=== SENDING {len(messages)} MESSAGES TO LLM ===\n", file=sys.stderr)
        # Pretty print the messages for better debugging
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = msg.content[:1000] + "..." if len(msg.content) > 1000  else msg.content
            print(f"  {i+1}. {msg_type}: {content_preview}", file=sys.stderr)
        if user_message_added:
            print("Added new user message to the conversation", file=sys.stderr)
        else:
            print("No new user message added", file=sys.stderr)
        
        # Call the model to generate code
        response = model.invoke(messages)
        
        # Log the entire response message for debugging
        print(f"\n=== MODEL RESPONSE ===", file=sys.stderr)
        print(f"Message type: {type(response).__name__}", file=sys.stderr)
        print(f"Content: {response.content}", file=sys.stderr)
        print(f"===\n", file=sys.stderr)
        # Store the model's response
        messages.append(response)
        
        # Extract the first code block found in the response
        code = extract_and_combine_codeblocks(response.content)
        
        # Debug log
        print(f"\n=== MODEL GENERATED CODE ===\n{code}\n===\n", file=sys.stderr)
        
        # Check if this is the same code as last time (prevent infinite loops)
        is_repeated_code = False
        if state.script and code and state.script.strip() == code.strip():
            print("Model generated the same code as before - breaking potential infinite loop", file=sys.stderr)
            is_repeated_code = True
            
        # If coming from sandbox or if repeating the same code, force a non-code response
        # to break potential infinite loops
        if is_repeated_code:
            # Add a special message to force model to respond with text instead of code
            stop_msg = HumanMessage(content=
                "I notice you're generating the same code repeatedly. Please provide a text response "
                "summarizing what you've done so far instead of more code.")
            messages.append(stop_msg)
            
            # Log the number of messages being sent to the LLM for the text response
            print(f"\n=== SENDING 2 MESSAGES TO LLM (BREAKING LOOP) ===\n", file=sys.stderr)
            
            # Get a new response that should be text instead of code
            text_response = model.invoke([messages[-2], stop_msg])
            messages.append(text_response)
            
            # End the workflow to wait for next user input
            return {
                "messages": messages,
                "script": None,
                "next_node": END
            }
        
        # If thread issues were detected and we still got code, force a text response
        if browser_thread_issues and browser_error_count >= 2 and code:
            # Add a special message to force model to respond with text instead of code
            stop_msg = HumanMessage(content=
                "Please don't provide more code since we're experiencing browser thread issues. "
                "Instead, explain in text what you would do if the browser was working properly.")
            messages.append(stop_msg)
            
            # Get a new response that should be text instead of code
            text_response = model.invoke([messages[-2], stop_msg])
            messages.append(text_response)
            
            # End the workflow to wait for next user input
            return {
                "messages": messages,
                "script": None,
                "next_node": END
            }
        
        # Determine the next node
        # If model generated code, run it in sandbox
        # If no code, end to wait for next user input
        next_node = "sandbox" if code else END
        
        return {
            "messages": messages,
            "script": code,
            "next_node": next_node
        }

    def sandbox(state: CodeActState) -> dict:
        """Execute the generated code in a sandbox environment."""
        print("Executing code in sandbox...")
        # Get the code to execute
        code = state.script
        if not code:
            return {"next_node": END}
        
        # Get the current locals dict from context or create a new one
        locals_dict = state.context.get("__locals", {})
        
        # Execute the code
        print(f"\n=== EXECUTING CODE ===\n{code}\n===\n", file=sys.stderr)
        
        # Execute code directly without timeout
        try:
            output, new_vars = eval_fn(code, locals_dict)
        except Exception as e:
            # Handle execution errors
            error_msg = f"⚠️ ERROR: {type(e).__name__}: {str(e)}. Please fix and try again."
            output = error_msg
            new_vars = locals_dict.copy()
            
        print(f"\n=== CODE OUTPUT ===\n{output}\n===\n", file=sys.stderr)
        print(f"\n=== NEW VARS ===\n{new_vars}\n===\n", file=sys.stderr)
        
        # Update locals with new variables
        locals_dict.update(new_vars)
        
        # Update the context with the new locals
        context = state.context.copy()
        context["__locals"] = locals_dict
        
        # Clean the context to prevent recursion during serialization
        clean_context = clean_context_for_serialization(context)
        
        # Get browser page info if available
        page_info = ""
        try:
            # Check if browser variables are available
            if "browser_ready" in new_vars and new_vars.get("browser_ready", False):
                # Try to safely get the current URL
                if "last_url" in new_vars:
                    page_info += f"\nCurrent page URL: {new_vars['last_url']}\n"
                
                # Add other browser-specific information if available
                if "last_title" in new_vars:
                    page_info += f"Page title: {new_vars['last_title']}\n"
        except Exception:
            # Ignore errors when trying to get page info
            pass
            
        # Format variable information for the AI
        var_info = "\nAvailable variables:\n"
        important_vars = [k for k in new_vars.keys() if not k.startswith("__") and k not in 
                          ["browser_page", "browser_instance", "playwright_instance", "page", "browser"]]
        
        if important_vars:
            for var in important_vars:
                var_type = type(new_vars[var]).__name__
                var_preview = str(new_vars[var])
                if len(var_preview) > 100:
                    var_preview = var_preview[:100] + "..."
                var_info += f"- {var} ({var_type}): {var_preview}\n"
        else:
            var_info += "- No important variables defined yet\n"
        
        # Generate an AI message with the code output, page info, and variables
        output_msg = AIMessage(
            content=f"Code execution output:\n{output}\n{page_info}{var_info}\n is there anything else you need to do to solve the current task at hand or can you give the user an update? If there is something else you need to do, write the code to keep going until you solve the task at hand."
        )
        
        # Add the output message to the state
        messages = state.messages.copy()
        messages.append(output_msg)
        
        # Check if output contains error messages related to threading/greenlet
        if "Cannot switch to a different thread" in output or "greenlet.error" in output:
            # Special handling for thread-related errors
            special_msg = AIMessage(
                content="⚠️ The browser appears to be facing thread synchronization issues. Let's restart with a simpler approach that doesn't rely on complex browser interactions."
            )
            messages.append(special_msg)
        
        # Always return to model after code execution so it can react to the output
        return {
            "messages": messages,
            "context": clean_context,
            "next_node": "call_model",
            "script": None
        }

    def user_input(state: CodeActState) -> dict:
        """Handle the user input and decide what to do next."""
        # Check if we have user input
        if not state.last_user_input:
            # No user input to process, end the graph
            return {"next_node": END}
            
        # Go to call_model when we have user input
        return {"next_node": "call_model"}

    # Create a new graph with properly defined channels
    workflow = StateGraph(state_schema)
    
    # Add the next_node channel explicitly
    workflow.add_node("call_model", call_model)
    workflow.add_node("sandbox", sandbox)
    workflow.add_node("user_input", user_input)
    
    # Define conditional routing
    workflow.add_conditional_edges(
        "call_model",
        lambda state: state.next_node if hasattr(state, "next_node") and state.next_node else "sandbox" if state.script else END,
        {
            "sandbox": "sandbox",
            "user_input": "user_input",
            END: END
        }
    )
    
    # Add remaining edges
    workflow.add_edge("user_input", "call_model")
    workflow.add_edge("sandbox", "call_model")  # Always go from sandbox to call_model
    
    # Set the entry point
    workflow.set_entry_point("user_input")
    
    return workflow
    return workflow