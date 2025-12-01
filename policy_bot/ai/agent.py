import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from langchain.agents.react.base import DocstoreExplorer
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
from langgraph.checkpoint.memory import MemorySaver
import uuid
import sqlite3
import math
from collections import defaultdict, Counter
import re

from rank_bm25 import BM25Okapi

class BM25:
    """BM25 document ranking using rank_bm25 package"""

    def __init__(self, documents: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search for relevant documents using BM25"""
        query_words = self._tokenize(query.lower())
        scores = self.bm25.get_scores(query_words)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i], self.documents[i]) for i in top_indices]

class PolicyAgent:
    """
    A React agent specialized for answering company policy questions.
    Uses reasoning and acting to provide accurate policy information.
    """
    
    def __init__(self, model_name: str = "openai/gpt-4.1-mini", temperature: float = 0.1, 
                 thread_id: str = None, enable_database_search: bool = False, 
                 database_path: str = None, company_name: str = "Ozymandias Co",
                 documents: List[str] = None, additional_system_prompt: str = None):
        """
        Initialize the Policy Agent
        
        Args:
            model_name: The model to use via OpenRouter
            temperature: Model temperature for consistency
            thread_id: Thread ID for conversation memory
            enable_database_search: Whether to enable the BM25 database search tool
            database_path: Path to SQLite database for BM25 search
            company_name: Name of the company for policy assistance
            documents: List of document strings to create BM25 database with
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        if not thread_id:
            self.thread_id = str(uuid.uuid4())
        else:
            self.thread_id = thread_id
        
        self.enable_database_search = enable_database_search
        self.database_path = database_path
        self.bm25_engine = None
        self.company_name = company_name
        self.documents = documents
        self.additional_system_prompt = additional_system_prompt
        
        # Initialize BM25 search if enabled
        if self.enable_database_search:
            self._initialize_bm25_search()
        
        # System prompt for the policy agent
        self.system_prompt = f"""You are an expert Policy Agent named Art (short for Arthur). Your role is to help users understand and navigate {self.company_name}'s policies, rules, and procedures.

CORE RESPONSIBILITIES:
- Answer questions about {self.company_name} policies accurately and helpfully
- Provide clear guidance on company policies, procedures, and terms of service
- Help resolve disputes and conflicts based on {self.company_name}'s terms of service
- Explain user rights and responsibilities
- Offer practical solutions to common policy-related issues

GUIDELINES:
- Always be helpful, professional, and empathetic
- Provide accurate information based on {self.company_name}'s official policies
- When uncertain, acknowledge limitations and suggest contacting {self.company_name} support
- Break down complex policies into clear, actionable steps
- Consider all stakeholder perspectives when applicable
- Prioritize safety and security in all recommendations

RESPONSE FORMAT:
- Start with a brief, direct answer
- Provide detailed explanation when needed
- Include relevant policy references
- Offer practical next steps or alternatives
- Use clear, friendly language

Remember: You are here to help users navigate {self.company_name} policies effectively and resolve their concerns.

{self.additional_system_prompt}""".strip()

    def _initialize_bm25_search(self):
        """Initialize the BM25 search engine with database content"""
        self.bm25_engine = BM25(self.documents)
   
    def create_tools(self) -> List[BaseTool]:
        """Create tools for the React agent"""
        
        tools = []
        
        # Add BM25 database search tool if enabled
        if self.enable_database_search and self.bm25_engine:
            @tool
            def bm25_database_search(keywords: str) -> str:
                """Search the database using BM25 algorithm with keywords to find relevant policy information"""
                try:
                    results = self.bm25_engine.search(keywords, top_k=3)
                    if not results:
                        return "No relevant documents found in the database."
                    
                    response = f"Found {len(results)} relevant documents for '{keywords}':\n\n"
                    for i, (doc_id, score, content) in enumerate(results, 1):
                        response += f"{i}. (Score: {score:.2f}) {content}\n\n"
                    
                    return response
                except Exception as e:
                    return f"Error searching database: {str(e)}"
            
            tools.append(bm25_database_search)
        
        return tools

    def _create_react_agent(self):
        """Create the React agent with tools and prompt"""
        tools = self.create_tools()

        # Create the agent
        checkpointer = MemorySaver()  # For conversation memory
        agent = create_react_agent(
            self.llm, 
            tools, 
            prompt=self.system_prompt,
            checkpointer=checkpointer
        )
        return agent

    def ask_question(self, question: str, include_tool_calls: bool = False) -> Dict[str, Any]:
        """
        Ask a question to the Policy Agent
        
        Args:
            question: The user's question about company policies
            include_tool_calls: Whether to include tool calls and responses in the return
            
        Returns:
            Dictionary containing:
            - 'response': The agent's final response
            - 'tool_calls': List of tool calls made (if include_tool_calls=True)
            - 'tool_responses': List of tool responses (if include_tool_calls=True)
        """
        try:
            # Get the agent
            agent = self._create_react_agent()
            
            # Run the agent
            response = agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=question)
                    ]
                },
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            result = {
                "response": response['messages'][-1].content
            }
            
            if include_tool_calls:
                tool_calls = []
                tool_responses = []
                
                # Parse messages to extract tool calls and responses
                for message in response['messages']:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        # This is an AI message with tool calls
                        for tool_call in message.tool_calls:
                            tool_calls.append({
                                "tool_name": tool_call.get("name", "unknown"),
                                "tool_args": tool_call.get("args", {}),
                                "tool_call_id": tool_call.get("id", "unknown")
                            })
                    elif hasattr(message, 'tool_call_id'):
                        # This is a tool response message
                        tool_responses.append({
                            "tool_call_id": message.tool_call_id,
                            "content": message.content
                        })
                
                result["tool_calls"] = tool_calls
                result["tool_responses"] = tool_responses
            
            return result
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your question: {str(e)}. Please try rephrasing your question or contact {self.company_name} support directly."
            return {
                 "response": error_msg,
                 "tool_calls": [] if include_tool_calls else None,
                 "tool_responses": [] if include_tool_calls else None
             }
    
    def ask_question_simple(self, question: str) -> str:
        """
        Ask a question to the Policy Agent (simple version)
        
        Args:
            question: The user's question about company policies
            
        Returns:
            The agent's response as a string
        """
        result = self.ask_question(question, include_tool_calls=False)
        return result["response"]
