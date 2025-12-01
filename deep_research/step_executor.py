import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from datetime import datetime
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate  # Use langchain_core as in file_context_0
import re
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Initialize the SERP API search tool
def search_with_serpapi(query: str) -> str:
    """Search the web for the query using SerpApi."""
    print(f'🔍 Looking up "{query}"')
    search = GoogleSearch({
        "q": query,
        "api_key": os.environ.get("SERP_API_KEY"),
        "location": "Austin,Texas",
        "hl": "en",
        "gl": "us"
    })
    result = search.get_dict()
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Extract organic results
    if "organic_results" in result:
        organic_results = result["organic_results"]
        formatted_results = []
        
        for idx, res in enumerate(organic_results[:3]):  # Limit to first 3 results
            title = res.get("title", "No title")
            snippet = res.get("snippet", "No snippet")
            link = res.get("link", "No link")
            formatted_results.append(f"{idx+1}. {title}\n   {snippet}\n   URL: {link}\n")
        
        return "\n".join(formatted_results)
    
    return "No results found."

# Initialize the Firecrawl scraping tool
def scrape_with_firecrawl(url_and_format: str) -> str:
    """Scrape a webpage using Firecrawl. Input should be 'URL|format' where format is 'markdown' or 'links'."""
    print(f"🔥 Scraping {url_and_format}")
    try:
        from firecrawl import FirecrawlApp
    except ImportError:
        print("🔥 Error: firecrawl package not found. Please run `pip install firecrawl-py`")
        return "Error: firecrawl package not found. Please run `pip install firecrawl-py`"
    
    # Parse input - expect format like "https://example.com|markdown" or just "https://example.com"
    if "|" in url_and_format:
        website_url, what_to_return = url_and_format.split("|", 1)
        website_url = website_url.strip()
        what_to_return = what_to_return.strip()
    else:
        website_url = url_and_format.strip()
        what_to_return = "markdown"
    
    if what_to_return not in ["markdown", "links"]:
        what_to_return = "markdown"
    
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        print("🔥 Error: FIRECRAWL_API_KEY environment variable is required")
        return "Error: FIRECRAWL_API_KEY environment variable is required"
    
    try:
        firecrawl = FirecrawlApp(api_key=api_key)
        response = firecrawl.scrape_url(
            website_url, formats=['markdown', 'links']
        )
        response = getattr(response, what_to_return)
        
        if isinstance(response, str):
            # Replace data:image URLs with empty parentheses
            response = re.sub(r'\(data:image/.*?\)', '()', response)
        
        print(f"🔥 Scraped {url_and_format} and got {len(response)} characters")
        return response if response else "No content found"
    
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

# Create Tools
search_serp_tool = Tool(
    name="web_search",
    description="Google things for information using SerpApi. Input format: 'query' where query is the search query. Example: 'What is the weather in London?' and the result will be potentially relevant information from the web",
    func=search_with_serpapi
)

firecrawl_tool = Tool(
    name="web_crawler",
    description="Scrape a webpage using Firecrawl. Input format: 'URL|format' where format is 'markdown' (default) or 'links'. Example: 'https://example.com|markdown' and the result will be the entire website's content in markdown format",
    func=scrape_with_firecrawl
)

# Define the tools list
tools = [search_serp_tool, firecrawl_tool]

# Define a React Agent prompt using a template string, similar to file_context_0
def get_react_system_prompt() -> str:
    """Get the prompt template for the React agent with required input keys."""
    template = f"""\
Today is {datetime.now().strftime("%Y-%m-%d")}. You are a curious researcher.

You have access to the following tools:

{{tools}}
"""
    return template

react_prompt = ChatPromptTemplate.from_template(get_react_system_prompt())

# Choose the LLM that will drive the agent
def create_step_executor(openrouter_model_name):
    llm = ChatOpenAI(
        model=openrouter_model_name, base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    return create_agent(llm, tools, system_prompt=get_react_system_prompt(), checkpointer=MemorySaver())
    
