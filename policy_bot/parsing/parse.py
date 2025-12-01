import requests
from bs4 import BeautifulSoup
import os
from typing import List, Dict, Any, Set
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from urllib.parse import urljoin, urlparse
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class Scenario(BaseModel):
    """A single scenario with situation and solution"""
    situation: str = Field(description="The specific situation or scenario")
    solution: str = Field(description="The solution based on the article, with citation")

class ScenarioResponse(BaseModel):
    """Complete response with chain of thought and scenarios"""
    chain_of_thought: str = Field(description="Step-by-step reasoning process")
    scenarios: List[Scenario] = Field(description="List of scenarios with solutions")

class ScrapingResult(BaseModel):
    """Result of scraping operation"""
    url: str
    title: str
    content: str
    links_found: List[str]
    depth: int

# e.g. url = 'https://www.airbnb.com/help/article/41'
def scrape_article(url):
    '''
    Scrape an article from a url
    '''
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text().strip()

def find_related_links(url: str, base_domain: str = None, max_links: int = 10, patterns: List[str] = ['/help/', '/support/', '/article/', '/faq']) -> List[str]:
    """
    Find related links on a webpage, focusing on help/support articles
    
    Args:
        url: The URL to scrape for links
        base_domain: Optional base domain to filter links (e.g., 'airbnb.com')
        max_links: Maximum number of links to return
    
    Returns:
        List of related URLs
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
                
            # Convert relative URLs to absolute
            absolute_url = urljoin(url, href)
            
            if '?' in absolute_url:
                absolute_url = absolute_url.split('?')[0]
            
            if '#' in absolute_url:
                absolute_url = absolute_url.split('#')[0]
            
            # Parse the URL to check domain
            parsed_url = urlparse(absolute_url)

            
            # Filter links based on criteria
            if base_domain and base_domain not in parsed_url.netloc:
                continue
                
            # Look for help/support article patterns
            if any(pattern in absolute_url.lower() for pattern in patterns):
                # Avoid duplicate links
                if absolute_url not in links:
                    links.append(absolute_url)
                    
                    if len(links) >= max_links:
                        break
        
        logger.info(f"Found {len(links)} related links on {url}")
        return links
        
    except Exception as e:
        logger.error(f"Error finding links on {url}: {e}")
        return []

def scrape_article_with_metadata(url: str, find_links: bool = True, patterns: List[str] = ['/help/', '/support/', '/article/', '/faq']) -> ScrapingResult:
    """
    Scrape an article with additional metadata
    
    Args:
        url: The URL to scrape
        find_links: Whether to find related links (set to False to avoid duplicate link finding)
    
    Returns:
        ScrapingResult with content and metadata
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract main content
        content = soup.get_text()
        
        # Find related links only if requested
        links = find_related_links(url, patterns=patterns) if find_links else []
        
        return ScrapingResult(
            url=url,
            title=title,
            content=content,
            links_found=links,
            depth=0
        )
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return ScrapingResult(
            url=url,
            title="Error",
            content="",
            links_found=[],
            depth=0
        )

def recursive_scrape(
    start_urls: List[str], 
    max_depth: int = 3, 
    max_links_per_page: int = 5,
    base_domain: str = None,
    delay: float = 1.0,
    patterns: List[str] = ['/help/', '/support/', '/article/', '/faq']
) -> List[ScrapingResult]:
    """
    Recursively scrape articles starting from initial URLs
    
    Args:
        start_urls: List of URLs to start scraping from
        max_depth: Maximum recursion depth (default: 3)
        max_links_per_page: Maximum links to follow per page (default: 5)
        base_domain: Optional base domain filter (e.g., 'airbnb.com')
        delay: Delay between requests in seconds
    
    Returns:
        List of ScrapingResult objects
    """
    visited_urls: Set[str] = set()
    results: List[ScrapingResult] = []
    urls_to_process = [(url, 0) for url in start_urls]  # (url, depth)
    
    while urls_to_process:
        current_url, current_depth = urls_to_process.pop(0)
        
        # Skip if already visited or max depth reached
        if current_url in visited_urls or current_depth > max_depth:
            continue
            
        logger.info(f"Scraping {current_url} (depth: {current_depth})")
        
        # Add to visited set
        visited_urls.add(current_url)
        
        # Scrape the current URL (don't find links here, we'll do it separately)
        result = scrape_article_with_metadata(current_url, find_links=False, patterns=patterns)
        result.depth = current_depth
        results.append(result)
        
        # Add related links for next level (if not at max depth)
        if current_depth < max_depth:
            related_links = find_related_links(current_url, base_domain, max_links_per_page)
            for link in related_links:
                if link not in visited_urls:
                    urls_to_process.append((link, current_depth + 1))
        
        # Be respectful with delays
        time.sleep(delay)
    
    logger.info(f"Completed scraping. Total pages scraped: {len(results)}")
    return results

def generate_scenarios_from_text(article_text: str, model_name: str = "openai/gpt-4.1", k: int = 3) -> ScenarioResponse:
    """
    Generate 3 fictitious scenarios based on article text using LangChain with OpenRouter
    
    Args:
        article_text: The text content from an article
        model_name: The model to use via OpenRouter
    
    Returns:
        ScenarioResponse with chain of thought and k scenarios
    """
    
    # Initialize the language model
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    # Create the prompt template
    scenario_prompt = ChatPromptTemplate.from_template(
        """You are an expert at creating realistic scenarios based on policy and help articles.

Article Content:
{article_text}

Your task is to:
1. First, provide a chain of thought explaining your reasoning process
2. Generate exactly {k} realistic, fictitious scenarios that can be answered using ONLY the information provided in the article
3. For each scenario, provide a complete solution that cites the source article

Requirements:
- Scenarios should be realistic situations someone might encounter
- Solutions must be based entirely on information from the provided article
- Each solution should reference/cite the source article
- Scenarios should be diverse and cover different aspects of the article content

Please provide your response in the following format:

**Chain of Thought:**
[Your step-by-step reasoning process for creating these scenarios]

**Scenario 1:**
Situation: [Describe the specific situation]
Solution: [Provide the solution based on the article, with citation]

**Scenario 2:**
Situation: [Describe the specific situation]
Solution: [Provide the solution based on the article, with citation]

...

**Scenario {k}:**
Situation: [Describe the specific situation]
Solution: [Provide the solution based on the article, with citation]
"""
    )
    
    # Create the chain and get structured output
    chain = scenario_prompt | llm.with_structured_output(ScenarioResponse)
    
    # Generate the scenarios
    try:
        result = chain.invoke({"article_text": article_text, "k": k})
        return result
    except Exception as e:
        # Fallback to regular text output if structured output fails
        regular_chain = scenario_prompt | llm
        response = regular_chain.invoke({"article_text": article_text})
        
        # Parse the response manually if structured output fails
        content = response.content
        
        # Simple parsing - this could be improved with regex
        scenarios = []
        lines = content.split('\n')
        current_scenario = None
        chain_of_thought = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('**Chain of Thought:**'):
                continue
            elif line.startswith('**Scenario') and line.endswith(':**'):
                if current_scenario:
                    scenarios.append(current_scenario)
                current_scenario = {"situation": "", "solution": ""}
            elif line.startswith('Situation:'):
                if current_scenario:
                    current_scenario["situation"] = line.replace('Situation:', '').strip()
            elif line.startswith('Solution:'):
                if current_scenario:
                    current_scenario["solution"] = line.replace('Solution:', '').strip()
            elif '**Chain of Thought:**' in content and not chain_of_thought:
                # Extract chain of thought
                start = content.find('**Chain of Thought:**') + len('**Chain of Thought:**')
                end = content.find('**Scenario 1:**')
                if end != -1:
                    chain_of_thought = content[start:end].strip()
        
        if current_scenario:
            scenarios.append(current_scenario)
        
        return ScenarioResponse(
            chain_of_thought=chain_of_thought or "Generated scenarios based on article content analysis.",
            scenarios=[Scenario(**scenario) for scenario in scenarios[:3]]
        )
