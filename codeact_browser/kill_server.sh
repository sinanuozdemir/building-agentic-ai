#!/bin/bash

PORT=8000
echo "Looking for processes using port $PORT..."

# Try to find the PID using lsof
if command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -ti:$PORT)
    if [ -n "$PID" ]; then
        echo "Found process $PID using port $PORT"
        echo "Killing process..."
        kill $PID
        sleep 1
        
        # Check if process was killed
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process didn't terminate gracefully. Force killing..."
            kill -9 $PID
            sleep 1
        fi
        
        # Final check
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✓ Process successfully killed!"
        else
            echo "⚠️ Failed to kill process $PID"
            exit 1
        fi
    else
        echo "No process found using port $PORT"
    fi
# Fallback to netstat if lsof is not available
elif command -v netstat >/dev/null 2>&1; then
    PID=$(netstat -nlp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$PID" ]; then
        echo "Found process $PID using port $PORT"
        echo "Killing process..."
        kill $PID
        sleep 1
        
        # Check if process was killed
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process didn't terminate gracefully. Force killing..."
            kill -9 $PID
            sleep 1
        fi
        
        # Final check
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✓ Process successfully killed!"
        else
            echo "⚠️ Failed to kill process $PID"
            exit 1
        fi
    else
        echo "No process found using port $PORT"
    fi
else
    echo "⚠️ Error: Neither lsof nor netstat is available."
    echo "Please install one of these utilities to use this script."
    exit 1
fi

echo "Port $PORT should now be free to use." 