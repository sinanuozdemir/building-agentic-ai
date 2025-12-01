#!/usr/bin/env python3
"""
Research MCP Server with Firecrawl and SerpAPI tools
This server provides web search and scraping capabilities for research tasks.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Tool, TextContent
import mcp.types as types

# Import required libraries
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# Create server instance
server = Server("research-mcp-server")

def search_with_serpapi(query: str) -> str:
    """Search the web for the query using SerpApi."""
    if not SERPAPI_AVAILABLE:
        return "Error: SerpAPI package not available. Please install google-search-results"
    
    api_key = os.environ.get("SERP_API_KEY")
    if not api_key:
        return "Error: SERP_API_KEY environment variable is required"
    
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": api_key,
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
    
    except Exception as e:
        return f"Error performing search: {str(e)}"

def scrape_with_firecrawl(url: str, format_type: str = "markdown") -> str:
    """Scrape a webpage using Firecrawl."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl package not available. Please install firecrawl-py"
    
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return "Error: FIRECRAWL_API_KEY environment variable is required"
    
    if format_type not in ["markdown", "links"]:
        format_type = "markdown"
    
    try:
        firecrawl = FirecrawlApp(api_key=api_key)
        response = firecrawl.scrape_url(
            url, formats=['markdown', 'links']
        )
        response = getattr(response, format_type)
        
        if isinstance(response, str):
            # Replace data:image URLs with empty parentheses
            response = re.sub(r'\(data:image/.*?\)', '()', response)
        
        return response if response else "No content found"
    
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available research tools."""
    tools = []
    
    # Add SerpAPI search tool if available
    if SERPAPI_AVAILABLE:
        tools.append(Tool(
            name="web_search",
            description="Search the web for information using SerpApi. Returns top 3 search results with titles, snippets, and URLs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    }
                },
                "required": ["query"]
            }
        ))
    
    # Add Firecrawl scraping tool if available
    if FIRECRAWL_AVAILABLE:
        tools.append(Tool(
            name="scrape_website",
            description="Scrape a webpage using Firecrawl. Can return content in markdown format or extract links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "links"],
                        "description": "Format to return content in (default: markdown)",
                        "default": "markdown"
                    }
                },
                "required": ["url"]
            }
        ))
    
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls for research operations."""
    
    if name == "web_search":
        query = arguments.get("query")
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        result = search_with_serpapi(query)
        return [types.TextContent(type="text", text=result)]
    
    elif name == "scrape_website":
        url = arguments.get("url")
        format_type = arguments.get("format", "markdown")
        
        if not url:
            return [types.TextContent(type="text", text="Error: URL is required")]
        
        result = scrape_with_firecrawl(url, format_type)
        return [types.TextContent(type="text", text=result)]
   
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main function to run the MCP server."""
    # Import here to avoid issues with event loop
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="research-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 