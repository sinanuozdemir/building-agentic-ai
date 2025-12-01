import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Import from our graph module
from .rag_graph import RetrievedDocument

known_embedders = {
    "text-embedding-3-small": OpenAIEmbeddings,
    "text-embedding-3-large": OpenAIEmbeddings,
    "text-embedding-ada-002": OpenAIEmbeddings,
    "embed-english-v3.0": CohereEmbeddings,
    "embed-v4.0": CohereEmbeddings,
    "sentence-transformers/all-MiniLM-L6-v2": HuggingFaceEmbeddings,
    "jinaai/jina-embeddings-v3": HuggingFaceEmbeddings,
}

class RAGDatabase:
    """Database utilities for RAG system"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chroma_persist_directory: str = "./chromadb",
        collection_name: str = "documents",
        distance_function: str = "cosine"  # New parameter for distance function
    ):
        """
        Initialize the RAG database
        
        Args:
            embedding_model: Name of the embedding model
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            distance_function: Distance function for similarity search ("cosine", "l2", "ip")
        """
        self.collection_name = collection_name
        self.distance_function = distance_function
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        # Get the appropriate embedder class from known_embedders
        embedder_class = known_embedders.get(embedding_model)
        if not embedder_class:
            raise ValueError(f"Unknown embedding model: {embedding_model}. Supported models: {list(known_embedders.keys())}")
        # Initialize the embedder with appropriate parameters
        if embedder_class == HuggingFaceEmbeddings:
            self.embeddings = embedder_class(model_name=embedding_model, model_kwargs={"trust_remote_code": True})
        else:
            self.embeddings = embedder_class(model=embedding_model)
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Set up collection metadata for distance function
        collection_metadata = {"hnsw:space": distance_function}
        
        # Initialize vector store with proper distance function configuration
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            collection_metadata=collection_metadata
        )
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the vector store"""
        print(f"📝 Adding {len(documents)} documents to vector store...")
        
        # Create Document objects
        docs = []
        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append(Document(page_content=doc_text, metadata=metadata))
        
        # Add to vector store
        self.vector_store.add_documents(docs)
        print("✅ Documents added successfully")
    
    def add_documents_from_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Add documents from a text file, splitting into chunks"""
        print(f"📄 Loading documents from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking strategy
        chunks = []
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Add metadata with source file
        metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]
        
        self.add_documents(chunks, metadatas)
        print(f"✅ Added {len(chunks)} chunks from {file_path}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection"""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {
                "error": str(e),
                "document_count": 0
            }
    
    def search_documents(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Search for relevant documents"""
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        # Convert to RetrievedDocument objects
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append(RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                # score=1-score if self.distance_function == "cosine" else score  # Chromadb returns 1-cosine similarity
                score=score
            ))
        
        return retrieved_docs
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            # Recreate the collection with the same distance function
            collection_metadata = {"hnsw:space": self.distance_function}
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                collection_metadata=collection_metadata
            )
            print(f"🗑️ Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")


class QueryResultsManager:
    """Manages query results and CSV export functionality"""
    
    def __init__(self):
        self.query_history = []
    
    def add_query_result(self, result: Dict[str, Any]):
        """Add a query result to the history"""
        self.query_history.append(result)
    
    def export_query_history_to_csv(self, filename: Optional[str] = None, output_dir: str = "rag_exports") -> str:
        """Export all query results to CSV format"""
        if not self.query_history:
            print("❌ No query history to export")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_query_results_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create full file path
        file_path = os.path.join(output_dir, filename)
        
        # Prepare data for CSV
        csv_data = []
        
        for query_result in self.query_history:
            query = query_result["query"]
            answer = query_result["answer"]
            timestamp = query_result["timestamp"]
            model_name = query_result["model_name"]
            embedding_model = query_result["embedding_model"]
            k_value = query_result["k"]
            
            # Create one row per retrieved document
            for i, doc in enumerate(query_result["retrieved_documents"]):
                csv_data.append({
                    "timestamp": timestamp,
                    "query": query,
                    "model_name": model_name,
                    "embedding_model": embedding_model,
                    "k_value": k_value,
                    "document_rank": i,
                    "document_score": doc.score,
                    "document_content": doc.content,
                    "document_metadata": json.dumps(doc.metadata) if doc.metadata else "",
                    "final_answer": answer
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(file_path, index=False)
        
        print(f"📊 Exported {len(csv_data)} document retrievals from {len(self.query_history)} queries to: {file_path}")
        return file_path
   
    def clear_query_history(self):
        """Clear the query history"""
        self.query_history = []
        print("🗑️ Query history cleared")
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about the query history"""
        if not self.query_history:
            return {"total_queries": 0}
        
        total_docs = sum(len(q['retrieved_documents']) for q in self.query_history)
        all_scores = []
        all_answer_lengths = []
        
        for query_result in self.query_history:
            all_scores.extend([doc.score for doc in query_result["retrieved_documents"]])
            all_answer_lengths.append(len(query_result["answer"]))
        
        return {
            "total_queries": len(self.query_history),
            "total_documents_retrieved": total_docs,
            "avg_documents_per_query": total_docs / len(self.query_history),
            "avg_relevance_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "avg_answer_length": sum(all_answer_lengths) / len(all_answer_lengths) if all_answer_lengths else 0,
            "best_relevance_score": min(all_scores) if all_scores else None,
            "worst_relevance_score": max(all_scores) if all_scores else None
        }
