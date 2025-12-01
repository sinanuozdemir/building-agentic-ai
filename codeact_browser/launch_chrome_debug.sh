#!/bin/bash

# Script to launch Chrome browser with remote debugging enabled
# This makes it available for use with the CodeAct browser automation

echo "Launching Chrome browser with remote debugging enabled on port 9222..."
echo "This will allow the CodeAct browser automation to connect to it."

# Check if Chrome is installed
if [ ! -d "/Applications/Google Chrome.app" ]; then
  echo "Error: Chrome browser not found in Applications folder."
  echo "If it's installed elsewhere, modify this script accordingly."
  exit 1
fi

# Launch Chrome with remote debugging enabled
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

echo "Chrome browser launched with debugging enabled."
echo "In the CodeAct app, check 'Connect to existing browser' and use endpoint: http://localhost:9222" 