from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
import sqlite3
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import describe_database

class SQLOutput(BaseModel):
    reasoning: str = Field(description="think through step by step how to solve the problem.")
    sql_query: str = Field(description="The executable sql query")

    def dict(self):
        return {
            "reasoning": self.reasoning,
            "sql_query": self.sql_query
        }

class SQLOutputReflection(BaseModel):
    reasoning: str = Field(description="think through step by step how to solve the problem.")
    first_draft_sql_query: str = Field(description="The first draft of the executable sql query")
    reflection: str = Field(description="reflect on the sql query and the reasoning and improve it if necessary.")
    sql_query: str = Field(description="The final executable sql query. This query will be executed against the database to get the answer.")

    def dict(self):
        return {
            "reasoning": self.reasoning,
            "first_draft_sql_query": self.first_draft_sql_query,
            "reflection": self.reflection,
            "sql_query": self.sql_query
        }

class NoReasoningSQLOutput(BaseModel):
    sql_query: str = Field(description="The executable sql query")

    def dict(self):
        return {
            "reasoning": self.reasoning,
            "sql_query": self.sql_query
        }

class NoReasoningSQLOutputReflection(BaseModel):
    first_draft_sql_query: str = Field(description="The first draft of the executable sql query")
    reflection: str = Field(description="reflect on the sql query and the reasoning and improve it if necessary.")
    sql_query: str = Field(description="The final executable sql query. This query will be executed against the database to get the answer.")
    
    def dict(self):
        return {
            "first_draft_sql_query": self.first_draft_sql_query,
            "reflection": self.reflection,
            "sql_query": self.sql_query
        }

class TextToSQL(BaseModel):
    question: str
    evidence: Optional[str]
    expected_sql: Optional[str]
    difficulty: str
    db_name: str
    db_path: Optional[str] = None  # Constructor variable for database path
    
    # Metadata fields for tracking LLM usage and performance
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    model_name_used: Optional[str] = None
    finish_reason: Optional[str] = None
    llm_response_id: Optional[str] = None
    
    # Cost tracking fields
    actual_cost: Optional[float] = None  # Real cost from API in micro-units
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    def _llm_call(
        self,
        prompt,
        model_name='gpt-4.1-mini',
        structured_output=SQLOutput,
        **inference_kwargs
    ):
        if 'temperature' not in inference_kwargs:
            inference_kwargs['temperature'] = 0.7
        
        # Enable usage accounting to get real-time cost information
        if 'extra_body' not in inference_kwargs:
            inference_kwargs['extra_body'] = {}
        inference_kwargs['extra_body']['usage'] = {"include": True}
        
        client = ChatOpenAI(
            model=model_name,
            **inference_kwargs
        ).with_structured_output(structured_output, include_raw=True)

        # Measure latency
        start_time = time.time()
        response = client.invoke(prompt)
        end_time = time.time()
        
        # Store metadata
        self.latency_seconds = end_time - start_time
        
        if 'raw' in response and hasattr(response['raw'], 'response_metadata'):
            metadata = response['raw'].response_metadata
            
            # Extract token usage information
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
                self.input_tokens = token_usage.get('prompt_tokens')
                self.output_tokens = token_usage.get('completion_tokens')
                self.total_tokens = token_usage.get('total_tokens')
                
                # Extract real cost information if available
                if 'cost' in token_usage:
                    # Cost is already in dollars, not micro-units
                    self.actual_cost = token_usage.get('cost')
                
                # Extract detailed token information
                if 'prompt_tokens_details' in token_usage:
                    prompt_details = token_usage['prompt_tokens_details']
                    self.cached_tokens = prompt_details.get('cached_tokens', 0)
                
                if 'completion_tokens_details' in token_usage:
                    completion_details = token_usage['completion_tokens_details']
                    self.reasoning_tokens = completion_details.get('reasoning_tokens', 0)
            
            # Store other useful metadata
            self.model_name_used = metadata.get('model_name', model_name)
            self.finish_reason = metadata.get('finish_reason')
            self.llm_response_id = metadata.get('id')

        return response

    def path_to_db(self):
        # Use constructor variable if provided, otherwise use default pattern
        if self.db_path:
            return self.db_path
        return f"../../dbs/dev_databases/{self.db_name}/{self.db_name}.sqlite"

    def _run_sql_against_db(self, query):
        if not query:
            raise ValueError('query is required')
        conn = sqlite3.connect(self.path_to_db())
        cursor = conn.cursor()
        cursor.execute(query)

        resulting_rows = [row for row in cursor.fetchall()]
        conn.close()
        return resulting_rows

    def expected_answer(self):
        return self._run_sql_against_db(self.expected_sql)
    
    def get_metadata(self):
        """Return a dictionary of all metadata collected during LLM calls."""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'latency_seconds': self.latency_seconds,
            'model_name_used': self.model_name_used,
            'finish_reason': self.finish_reason,
            'llm_response_id': self.llm_response_id,
            'actual_cost': self.actual_cost,
            'cached_tokens': self.cached_tokens,
            'reasoning_tokens': self.reasoning_tokens
        }
    
    def print_metadata(self):
        """Print metadata in a formatted way."""
        metadata = self.get_metadata()
        print("\n📊 LLM Call Metadata:")
        print(f"   🔸 Model: {metadata['model_name_used']}")
        print(f"   🔸 Input tokens: {metadata['input_tokens']}")
        print(f"   🔸 Output tokens: {metadata['output_tokens']}")
        print(f"   🔸 Total tokens: {metadata['total_tokens']}")
        
        # Show detailed token breakdown if available
        if metadata['cached_tokens'] is not None:
            print(f"   🔸 Cached tokens: {metadata['cached_tokens']}")
        if metadata['reasoning_tokens'] is not None:
            print(f"   🔸 Reasoning tokens: {metadata['reasoning_tokens']}")
        
        print(f"   🔸 Latency: {metadata['latency_seconds']:.3f} seconds")
        print(f"   🔸 Finish reason: {metadata['finish_reason']}")
        print(f"   🔸 Response ID: {metadata['llm_response_id']}")
        
        # Show actual cost if available
        if metadata['actual_cost'] is not None:
            # Cost is already in dollars from the API
            print(f"   💰 Actual cost: ${metadata['actual_cost']:.6f}")
        
        if metadata['total_tokens'] and metadata['latency_seconds']:
            tokens_per_second = metadata['total_tokens'] / metadata['latency_seconds']
            print(f"   🔸 Throughput: {tokens_per_second:.1f} tokens/second")
