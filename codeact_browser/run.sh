#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install or upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
if command_exists playwright; then
    python -m playwright install --with-deps
else
    echo "Playwright doesn't seem to be installed correctly. Retrying installation..."
    pip install playwright
    python -m playwright install --with-deps
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "Please edit the .env file and add your OpenAI API key before running the application."
    exit 1
fi

# Run Streamlit
echo "Starting Streamlit application..."
streamlit run test_codeact.py 