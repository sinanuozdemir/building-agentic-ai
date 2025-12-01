#!/bin/bash

# Twilio FastAPI Voice Assistant Setup Script
# This script sets up and runs the FastAPI server with ngrok tunneling

set -e  # Exit on any error

echo "Setting up Twilio FastAPI Voice Assistant..."

# Configuration
DOMAIN="twilio-applied-ai.ngrok.io"  # Your custom ngrok domain
PORT=5015
WEBHOOK_ENDPOINT="/incoming-call"
MEDIA_ENDPOINT="/media-stream"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to cleanup processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    echo "Stopping ngrok..."
    pkill -f ngrok || true
    echo "Stopping Flask app..."
    pkill -f "uvicorn" || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check for required commands
if ! command_exists ngrok; then
    echo -e "${RED}Error: ngrok is not installed. Please install ngrok first.${NC}"
    echo "Visit: https://ngrok.com/download"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Requirements installed successfully!${NC}"
else
    echo -e "${RED}Failed to install requirements!${NC}"
    exit 1
fi

# Check ngrok authentication
echo "Checking ngrok authentication..."
if ! ngrok config check >/dev/null 2>&1; then
    echo -e "${RED}ngrok is not authenticated. Please run 'ngrok authtoken YOUR_TOKEN' first.${NC}"
    exit 1
fi

# Kill any existing ngrok processes
pkill -f ngrok || true
sleep 2

# Start ngrok tunnel
echo "Starting ngrok tunnel..."
ngrok http $PORT --url=$DOMAIN --region=eu --log=stdout > ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to start
sleep 3

# Check if ngrok started successfully
if ! ps -p $NGROK_PID > /dev/null 2>&1; then
    echo -e "${RED}Failed to start ngrok. Check ngrok.log for details.${NC}"
    echo "Common issues:"
    echo "1. Domain '$DOMAIN' might not be available"
    echo "2. You might need a paid ngrok account for custom domains"
    echo "3. Check ngrok.log for detailed error information"
    cat ngrok.log
    exit 1
fi

# Extract URLs from ngrok
sleep 2
PUBLIC_URL="https://$DOMAIN"
LOCAL_URL="http://localhost:$PORT"
WEBHOOK_URL="$PUBLIC_URL$WEBHOOK_ENDPOINT"
WEBSOCKET_URL="wss://$DOMAIN$MEDIA_ENDPOINT"

echo -e "${GREEN}ngrok tunnel started successfully!${NC}"
echo "Public URL: $PUBLIC_URL"
echo "Local URL: $LOCAL_URL"
echo "Webhook URL: $WEBHOOK_URL"
echo "WebSocket endpoint: $WEBSOCKET_URL"
echo "Recordings: $PUBLIC_URL/recordings"
echo ""

# Display setup instructions
echo -e "${GREEN}=== SETUP INSTRUCTIONS ===${NC}"
echo "1. Configure your Twilio number webhook to:"
echo "   $WEBHOOK_URL"
echo "2. Set the HTTP method to POST"
echo "3. Call your Twilio number to test!"
echo "4. View recordings at: $PUBLIC_URL/recordings"
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY environment variable is not set.${NC}"
    echo "Text-to-speech functionality will not work without it."
    echo "Set it with: export OPENAI_API_KEY=your-api-key"
    echo ""
fi

# Start the FastAPI application
echo -e "${GREEN}Starting FastAPI app...${NC}"
echo "Press Ctrl+C to stop both servers"
echo ""

# Start uvicorn in the background and capture its PID
uvicorn twilio_app:app --host 0.0.0.0 --port $PORT --reload &
UVICORN_PID=$!

# Wait for FastAPI to start
sleep 2

# Check if FastAPI started successfully
if ! ps -p $UVICORN_PID > /dev/null 2>&1; then
    echo -e "${RED}Failed to start FastAPI server!${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}FastAPI server started successfully!${NC}"
echo "Server is running at: $LOCAL_URL"
echo "API documentation at: $LOCAL_URL/docs"
echo ""

# Keep the script running and wait for user to stop
wait $UVICORN_PID 