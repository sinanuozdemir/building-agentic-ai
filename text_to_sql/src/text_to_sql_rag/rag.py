import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated
from dataclasses import dataclass
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Vector store imports
from langchain_chroma import Chroma

# Import our separated components
from .rag_graph import RAGWorkflow, RAGState, RetrievedDocument
from .rag_db import RAGDatabase, QueryResultsManager


@dataclass
class RetrievedDocument:
    """Represents a document retrieved from the vector store"""
    content: str
    metadata: Dict[str, Any]
    score: float

class RAGSystem:
    """Main RAG System that orchestrates the workflow and database components"""
    
    def __init__(
        self,
        model_name: str = "openai/gpt-4.1-nano",
        collection_name: str = "documents",
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        k: int = 5,
        distance_function: str = "cosine",  # New parameter for distance function
        skip_generation: bool = False,
        workflow: RAGWorkflow = None
    ):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the LLM model to use
            collection_name: Name of the ChromaDB collection
            openai_api_key: OpenAI API key
            base_url: Base URL for API calls (e.g., OpenRouter)
            k: Number of documents to retrieve
            distance_function: Distance function for similarity search ("cosine", "l2", "ip")
        """
        self.model_name = model_name
        self.k = k
        self.distance_function = distance_function
        self.skip_generation = skip_generation
        self.cohere_api_key = cohere_api_key
        
        
        # Initialize workflow
        self.workflow = workflow
        
        # Initialize results manager
        self.results_manager = QueryResultsManager()
    
    def query(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Query the RAG system and store detailed results"""
        print(f"\n🚀 Starting RAG query: {question}")
        
        # Create initial state
        initial_state = RAGState(
            query=question,
            skip_generation=self.skip_generation,
            k=k or self.k
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        # Create detailed result with ordered documents
        detailed_result = {
            "query": question,
            "answer": result["final_answer"],
            "retrieved_documents": result["retrieved_documents"],
            "context": result["context"],
            "messages": result["messages"],
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "embedding_model": self.workflow.db.embedding_model,
            "k": k or self.k
        }
        
        # Store in query history for CSV export
        self.results_manager.add_query_result(detailed_result)
        
        return detailed_result
    
    # Database methods (delegate to RAGDatabase)
    def add_documents(self, documents, metadatas=None):
        """Add documents to the vector store"""
        return self.workflow.db.add_documents(documents, metadatas)
    
    def add_documents_from_file(self, file_path, chunk_size=1000, chunk_overlap=200):
        """Add documents from a text file"""
        return self.workflow.db.add_documents_from_file(file_path, chunk_size, chunk_overlap)
    
    def get_collection_info(self):
        """Get information about the document collection"""
        return self.workflow.db.get_collection_info()
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        return self.workflow.db.clear_collection()
    
    # Results management methods (delegate to QueryResultsManager)
    def export_query_history_to_csv(self, filename=None, output_dir="rag_exports"):
        """Export all query results to CSV format"""
        return self.results_manager.export_query_history_to_csv(filename, output_dir)
    
    def clear_query_history(self):
        """Clear the query history"""
        return self.results_manager.clear_query_history()
    
    def get_query_stats(self):
        """Get statistics about the query history"""
        return self.results_manager.get_query_stats()
    
    @property
    def query_history(self):
        """Access to query history for backward compatibility"""
        return self.results_manager.query_history
