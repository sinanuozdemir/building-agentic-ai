from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing import List, Annotated

# LangChain imports
from langchain_openai import ChatOpenAI


@dataclass
class RetrievedDocument:
    """Represents a document retrieved from the vector store"""
    content: str
    metadata: Dict[str, Any]
    score: float


class RAGState(BaseModel):
    """State for the RAG workflow"""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    query: str = ""
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    context: str = ""
    final_answer: str = ""
    k: int = 5  # Number of documents to retrieve
    skip_generation: bool = False


class RAGWorkflow:
    """LangGraph workflow for RAG processing"""
    
    def __init__(self, llm: ChatOpenAI, db, distance_function: str = "cosine"):
        """
        Initialize the RAG workflow
        
        Args:
            llm: The language model to use
            db: The database for document retrieval
        """
        self.llm = llm
        self.db = db
        self.distance_function = distance_function
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for RAG"""
        
        def process_query(state: RAGState) -> Dict[str, Any]:
            """Process the incoming query and prepare for retrieval"""
            print(f"🔍 Processing query: {state.query}")
            
            # Add user message if not already present
            messages = state.messages.copy()
            if not any(isinstance(msg, HumanMessage) and msg.content == state.query for msg in messages):
                messages.append(HumanMessage(content=state.query))
            
            return {
                "messages": messages,
            }
        
        def retrieve_documents(state: RAGState) -> Dict[str, Any]:
            """Retrieve relevant documents from ChromaDB"""
            print(f"📚 Retrieving top {state.k} documents for query: {state.query}")
            
            try:
                # Perform similarity search
                results = self.db.vector_store.similarity_search_with_score(
                    query=state.query,
                    k=state.k
                )
                
                # Convert to RetrievedDocument objects
                retrieved_docs = []
                for doc, score in results:
                    retrieved_docs.append(RetrievedDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=1-score if self.distance_function == "cosine" else score  # Chromadb returns 1-cosine similarity
                    ))
                
                print(f"✅ Retrieved {len(retrieved_docs)} documents")
                for i, doc in enumerate(retrieved_docs, 1):
                    print(f"   {i}. Score: {doc.score:.4f} | Content: {doc.content[:100]}...")
                
                # Create context from retrieved documents
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    context_parts.append(f"Document {i}:\n{doc.content}")
                    if doc.metadata:
                        context_parts.append(f"Metadata: {doc.metadata}")
                    context_parts.append("")  # Empty line for separation
                
                context = "\n".join(context_parts)
                
                return {
                    "retrieved_documents": retrieved_docs,
                    "context": context,
                }
                
            except Exception as e:
                print(f"❌ Error retrieving documents: {e}")
                return {
                    "retrieved_documents": [],
                    "context": "No documents could be retrieved due to an error.",
                }
        
        def generate_answer(state: RAGState) -> Dict[str, Any]:
            """Generate answer using retrieved documents"""
            if state.skip_generation:
                return {
                    "messages": state.messages,
                    "final_answer": "Skipping answer generation",
                }
            
            print("🤖 Generating answer using retrieved context...")
            
            # Create the prompt template
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents.

Instructions:
1. Use the provided context documents to answer the user's question
2. If the answer can be found in the context, provide a comprehensive response
3. If the context doesn't contain enough information, clearly state what information is missing
4. Always cite which document(s) you're referencing when possible
5. Be accurate and don't make up information not present in the context

Context Documents:
{context}

Please answer the following question based on the context above."""

            user_prompt = "{query}"
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=system_prompt.format(context=state.context)),
                HumanMessage(content=user_prompt.format(query=state.query))
            ]
            
            try:
                # Generate response
                response = self.llm.invoke(messages)
                final_answer = response.content
                
                print(f"✅ Generated answer: {final_answer[:200]}...")
                
                # Add AI response to messages
                updated_messages = state.messages.copy()
                updated_messages.append(AIMessage(content=final_answer))
                
                return {
                    "messages": updated_messages,
                    "final_answer": final_answer,
                }
                
            except Exception as e:
                error_msg = f"Error generating answer: {e}"
                print(f"❌ {error_msg}")
                
                updated_messages = state.messages.copy()
                updated_messages.append(AIMessage(content=error_msg))
                
                return {
                    "messages": updated_messages,
                    "final_answer": error_msg,
                }
        
        # Create the workflow graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("process_query", process_query)
        workflow.add_node("retrieve_documents", retrieve_documents)
        workflow.add_node("generate_answer", generate_answer)
        
        # Add edges
        workflow.add_edge("process_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Set entry point
        workflow.set_entry_point("process_query")
        
        return workflow.compile()
    
    def invoke(self, initial_state: RAGState) -> Dict[str, Any]:
        """Run the workflow with the given initial state"""
        return self.workflow.invoke(initial_state) 