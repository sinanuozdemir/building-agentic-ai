#!/bin/bash

# Deep Research Assistant - Streamlit Launcher Script
echo "🔍 Deep Research Assistant - Streamlit Launcher"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "⚠️  Please run this script from the deep_research directory"
    echo "Current directory: $(pwd)"
    echo "Try: cd deep_research && ./run_streamlit.sh"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/lib/python*/site-packages/streamlit" ]; then
    echo "📦 Installing requirements..."
    pip install -r streamlit_requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Please create one with your API keys:"
    echo ""
    echo "OPENROUTER_API_KEY=your_openrouter_api_key_here"
    echo "FIRECRAWL_API_KEY=your_firecrawl_api_key_here"
    echo "SERP_API_KEY=your_serp_api_key_here"
    echo ""
    echo "See README_STREAMLIT.md for more information."
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Launch Streamlit
echo "🚀 Launching Streamlit app..."
echo "The app will open in your default browser"
echo "Use Ctrl+C to stop the app"
echo "=================================================="

streamlit run streamlit_app.py --server.port 8501 --server.address localhost 