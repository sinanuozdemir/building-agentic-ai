# Text-to-SQL RAG System Prototype

A Flask web application that demonstrates a Text-to-SQL RAG (Retrieval-Augmented Generation) system. This prototype allows users to ask natural language questions about various databases and get SQL queries generated through a RAG pipeline, with the results displayed in a modern web interface.

## Features

- **Database Selection**: Choose from multiple available SQLite databases
- **Natural Language Queries**: Ask questions in plain English
- **RAG Evidence Display**: See the retrieved evidence used to generate the SQL
- **SQL Generation**: View the generated SQL query with syntax highlighting
- **Query Execution**: Execute the SQL and see results in a formatted table
- **Error Handling**: Graceful handling of SQL errors and system failures
- **Responsive Design**: Modern Bootstrap-based UI that works on all devices

## Available Databases

The system includes several databases:
- California Schools
- Card Games
- Student Club
- Superhero
- Thrombosis Prediction
- Toxicology
- Financial
- Formula 1
- Codebase Community
- Debit Card Specializing
- European Football 2

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (set in your environment or .env file)

### Installation

1. **Navigate to the prototype directory**:
   ```bash
   cd text_to_sql/prototype
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the prototype directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Make sure the RAG system has data**:
   The system requires vector embeddings to be pre-populated. If this is the first run, you may need to run the data ingestion process from the parent directory first.

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Select a Database**: Choose from the dropdown list of available databases
2. **Enter a Question**: Type your natural language question or click the sample question
3. **Run Query**: Click "Run RAG Query" to process your request
4. **View Results**: The system will display:
   - Evidence retrieved from the vector store (with similarity scores)
   - Generated SQL query
   - Query execution results (or error messages)

## Example Queries

### California Schools Database
- "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
- "Which schools have the highest SAT scores?"
- "How many charter schools are there in Los Angeles?"

### Card Games Database
- "What are the names of all card games that can be played by exactly 4 players?"
- "Which games have the highest complexity rating?"

### Superhero Database
- "What are the names of all superheroes with blue eyes?"
- "Which superhero has the most powers?"

## System Architecture

The application consists of:

1. **Flask Backend** (`app.py`):
   - REST API endpoints for database operations
   - RAG system integration
   - SQL query execution
   - Error handling

2. **Frontend** (`templates/index.html`):
   - Bootstrap-based responsive UI
   - JavaScript for dynamic interactions
   - Real-time result display

3. **RAG Integration**:
   - Uses the existing `text_to_sql_rag` system
   - ChromaDB for vector storage
   - OpenAI embeddings and language models

## API Endpoints

- `GET /` - Main application interface
- `GET /api/databases` - List available databases
- `GET /api/sample-question/<database>` - Get sample question for database
- `POST /api/query` - Execute RAG query and SQL

## Error Handling

The system gracefully handles:
- Invalid SQL queries
- Database connection errors
- RAG system failures
- Missing vector embeddings
- API rate limits

## Development

To modify or extend the system:

1. **Backend changes**: Edit `app.py`
2. **Frontend changes**: Edit `templates/index.html`
3. **Database queries**: The system automatically detects available databases in the `../dbs/dev_databases/` directory
4. **RAG configuration**: Modify the RAG system parameters in the `query_rag()` function

## Troubleshooting

### Common Issues

1. **No databases found**: Ensure the database files exist in `../dbs/dev_databases/`
2. **RAG system errors**: Check your OpenAI API key and network connection
3. **Empty results**: The vector store may need to be populated with embeddings
4. **SQL errors**: The generated SQL may need refinement based on the specific database schema

### Logs

The application prints debug information to the console, including:
- RAG query processing status
- SQL execution details
- Error messages

## Contributing

To contribute to this prototype:
1. Follow the existing code style
2. Add error handling for new features
3. Update the README for any new functionality
4. Test with multiple databases 