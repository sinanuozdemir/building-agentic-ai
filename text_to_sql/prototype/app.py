import os
import sys
import sqlite3
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Add the src directory to the path so we can import our RAG system
# Check if we're in Docker (where src is mounted at /app/src) or running locally
if os.path.exists('/app/src'):
    # Docker environment
    sys.path.append('/app/src')
else:
    # Local environment
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_to_sql_rag.rag import RAGSystem
from text_to_sql_rag.rag_db import known_embedders

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
# Check if we're in Docker or local environment for database path
if os.path.exists('/app/dbs'):
    # Docker environment
    DATABASES_DIR = '/app/dbs/dev_databases'
    DEV_JSON_PATH = '/app/dev.json'
else:
    # Local environment  
    DATABASES_DIR = os.path.join(os.path.dirname(__file__), '..', 'dbs', 'dev_databases')
    DEV_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'dev.json')

# Sample questions for each database
SAMPLE_QUESTIONS = {
    "california_schools": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
    "card_games": "What are the names of all card games that can be played by exactly 4 players?",
    "student_club": "Which student has the highest GPA in the Computer Science club?",
    "superhero": "What are the names of all superheroes with blue eyes?",
    "thrombosis_prediction": "List all patients who were followed up at the outpatient clinic?",
    "toxicology": "What is the average LD50 value for all compounds tested?",
    "financial": "What is the total amount of all successful transactions?",
    "formula_1": "Who won the most Formula 1 races in 2021?",
    "codebase_community": "Which repository has the most commits?",
    "debit_card_specializing": "What is the total amount spent on groceries last month?",
    "european_football_2": "Which team scored the most goals in the 2015-2016 season?"
}

def get_available_databases():
    """Get list of available SQLite databases"""
    databases = []
    if os.path.exists(DATABASES_DIR):
        for db_dir in os.listdir(DATABASES_DIR):
            db_path = os.path.join(DATABASES_DIR, db_dir)
            if os.path.isdir(db_path):
                sqlite_file = os.path.join(db_path, f"{db_dir}.sqlite")
                if os.path.exists(sqlite_file):
                    databases.append(db_dir)
    return sorted(databases)

def get_available_embedders():
    """Get list of available embedding models"""
    return list(known_embedders.keys())

def load_evidence_data():
    """Load evidence queries from dev.json"""
    try:
        if not os.path.exists(DEV_JSON_PATH):
            print(f"⚠️ dev.json not found at {DEV_JSON_PATH}")
            return {}
        
        with open(DEV_JSON_PATH, 'r') as f:
            dev_data = json.load(f)
        
        # Extract evidence queries by database
        queries_by_db = {}
        for entry in dev_data:
            db_id = entry.get('db_id')
            evidence = entry.get('evidence', '')
            
            if db_id and evidence.strip():
                if db_id not in queries_by_db:
                    queries_by_db[db_id] = []
                
                # Split evidence by semicolons and clean
                evidence_items = [e.strip() for e in evidence.split(';') if e.strip()]
                queries_by_db[db_id].extend(evidence_items)
        
        return queries_by_db
    except Exception as e:
        print(f"❌ Error loading evidence data: {e}")
        return {}

def populate_chromadb_if_empty(rag_system, database):
    """Populate ChromaDB with evidence if it's empty"""
    try:
        # Check if collection has documents
        info = rag_system.get_collection_info()
        document_count = info.get('document_count', 0)
        
        if document_count > 0:
            print(f"✅ ChromaDB already has {document_count} documents for {database}")
            return
        
        print(f"📝 ChromaDB is empty for {database}, populating with evidence...")
        
        # Load evidence data
        evidence_by_db = load_evidence_data()
        evidence_list = evidence_by_db.get(database, [])
        # remove duplicates
        evidence_list = list(set(evidence_list))
        
        if not evidence_list:
            print(f"⚠️ No evidence found for database {database}")
            return
        
        # Create metadata for each evidence item
        metadatas = [{"database": database, "type": "evidence", "index": i} for i in range(len(evidence_list))]
        
        # Add evidence to vector store
        rag_system.add_documents(evidence_list, metadatas)
        
        new_info = rag_system.get_collection_info()
        print(f"✅ Added {len(evidence_list)} evidence items to ChromaDB for {database}")
        print(f"📊 Total documents now: {new_info.get('document_count', 0)}")
        
    except Exception as e:
        print(f"❌ Error populating ChromaDB: {e}")

def execute_sql_query(database_name, sql_query):
    """Execute SQL query on the specified database"""
    try:
        db_path = os.path.join(DATABASES_DIR, database_name, f"{database_name}.sqlite")
        if not os.path.exists(db_path):
            return {"error": f"Database {database_name} not found"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(sql_query)
        
        # Get column names
        columns = [description[0] for description in cursor.description] if cursor.description else []
        
        # Fetch results
        results = cursor.fetchall()
        
        conn.close()
        
        return {
            "success": True,
            "columns": columns,
            "data": results,
            "row_count": len(results)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """Main page with the interface"""
    databases = get_available_databases()
    embedders = get_available_embedders()
    return render_template('index.html', 
                         databases=databases, 
                         embedders=embedders,
                         sample_questions=SAMPLE_QUESTIONS)

@app.route('/api/databases')
def get_databases():
    """API endpoint to get available databases"""
    return jsonify(get_available_databases())

@app.route('/api/embedders')
def get_embedders():
    """API endpoint to get available embedding models"""
    return jsonify(get_available_embedders())

@app.route('/api/sample-question/<database>')
def get_sample_question(database):
    """API endpoint to get sample question for a database"""
    return jsonify({"question": SAMPLE_QUESTIONS.get(database, "What data is available in this database?")})

@app.route('/api/query', methods=['POST'])
def query_rag():
    """API endpoint to run RAG query and execute SQL"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        database = data.get('database', '')
        embedding_model = data.get('embedding_model', 'text-embedding-3-small')
        
        if not question or not database:
            return jsonify({"error": "Question and database are required"}), 400
        
        if embedding_model not in known_embedders:
            return jsonify({"error": f"Unknown embedding model: {embedding_model}"}), 400
        
        # Initialize RAG system for this database and embedding model
        collection_name = f"documents_{database}_{embedding_model.replace('/', '_').replace('-', '_')}"
        chroma_dir = os.path.join(os.path.dirname(__file__), 'chromadb', f"{database}_{embedding_model.replace('/', '_').replace('-', '_')}")
        
        rag_system = RAGSystem(
            model_name="gpt-4o-mini",
            embedding_model=embedding_model,
            chroma_persist_directory=chroma_dir,
            collection_name=collection_name,
            k=5,
            skip_generation=False
        )
        
        # Populate ChromaDB if empty
        populate_chromadb_if_empty(rag_system, database)
        
        # Run RAG query
        print(f"🚀 Running RAG query for database: {database}")
        print(f"🔧 Using embedding model: {embedding_model}")
        print(f"📝 Question: {question}")
        
        rag_result = rag_system.query(question)
        
        # Extract the SQL query from the RAG result
        sql_query = None
        final_answer = rag_result.get('answer', '')
        
        # Try to extract SQL from the final answer
        if '```sql' in final_answer.lower():
            # Extract SQL from markdown code blocks
            import re
            sql_match = re.search(r'```sql\s*(.*?)\s*```', final_answer, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_query = sql_match.group(1).strip()
        elif 'SELECT' in final_answer.upper():
            # Try to extract SQL that starts with SELECT
            import re
            sql_match = re.search(r'(SELECT.*?)(?:\n\n|\Z)', final_answer, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_query = sql_match.group(1).strip()
        
        # If no SQL found in answer, look in the context
        if not sql_query:
            context = rag_result.get('context', '')
            if 'SELECT' in context.upper():
                import re
                sql_match = re.search(r'(SELECT.*?)(?:\n|\Z)', context, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
        
        result = {
            "question": question,
            "database": database,
            "embedding_model": embedding_model,
            "rag_result": {
                "answer": final_answer,
                "context": rag_result.get('context', ''),
                "retrieved_documents": [
                    {
                        "content": doc.content,
                        "score": doc.score,
                        "metadata": doc.metadata
                    }
                    for doc in rag_result.get('retrieved_documents', [])
                ]
            },
            "sql_query": sql_query,
            "sql_result": None
        }
        
        # Execute SQL if we found one
        if sql_query:
            print(f"🗃️ Executing SQL: {sql_query}")
            sql_result = execute_sql_query(database, sql_query)
            result["sql_result"] = sql_result
        else:
            result["sql_result"] = {"error": "No SQL query found in the RAG response"}
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in query_rag: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 