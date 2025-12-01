#!/bin/bash

# Variables
SERVICE_PORT=8000
SERVICE_LOG="browser_service.log"

# Debug information
echo "=== Debug Information ===" > $SERVICE_LOG
echo "Current PATH: $PATH" >> $SERVICE_LOG
echo "Python location: $(which python)" >> $SERVICE_LOG
echo "Python version: $(python --version 2>&1)" >> $SERVICE_LOG
echo "PYTHONPATH: $PYTHONPATH" >> $SERVICE_LOG
echo "Current directory: $(pwd)" >> $SERVICE_LOG
echo "Listing installed packages:" >> $SERVICE_LOG
python -m pip freeze >> $SERVICE_LOG
echo "=== End Debug Information ===" >> $SERVICE_LOG
echo "" >> $SERVICE_LOG

# Ensure we're using the virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Add more debug info after activation
    echo "=== After venv activation ===" >> $SERVICE_LOG
    echo "Python location: $(which python)" >> $SERVICE_LOG
    echo "Python version: $(python --version 2>&1)" >> $SERVICE_LOG
    echo "=== End venv info ===" >> $SERVICE_LOG
    echo "" >> $SERVICE_LOG
fi

# Function to check if a process is running on a port
check_port() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -i:$1 >/dev/null 2>&1
        return $?
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln | grep -q ":$1 "
        return $?
    else
        echo "Cannot check port: neither lsof nor netstat is available"
        return 1
    fi
}

# Function to start the browser service
start_service() {
    echo "Starting browser service on port $SERVICE_PORT..."
    # Use python from the current environment
    echo "Running Python with path: $(which python)" >> $SERVICE_LOG
    
    # Run with explicit python path and append to log
    nohup $(which python) browser_service.py >> $SERVICE_LOG 2>&1 &
    SERVICE_PID=$!
    echo "Service started with PID $SERVICE_PID"
    
    # Wait for the service to become available
    echo "Waiting for service to start..."
    for i in {1..10}; do
        if curl -s http://localhost:$SERVICE_PORT/ >/dev/null; then
            echo "Browser service is up and running!"
            return 0
        fi
        echo "Waiting (attempt $i)..." >> $SERVICE_LOG
        sleep 1
    done
    
    echo "Failed to start browser service within timeout period. Check $SERVICE_LOG for details."
    kill $SERVICE_PID 2>/dev/null || true
    return 1
}

# Check if the service is already running
if check_port $SERVICE_PORT; then
    echo "Browser service is already running on port $SERVICE_PORT"
else
    start_service || exit 1
fi

echo "Browser service is running. You can now start the Streamlit app with:"
echo "streamlit run test_codeact.py" 