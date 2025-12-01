#!/bin/bash

# Script to launch Arc browser with remote debugging enabled
# This makes it available for use with the CodeAct browser automation

echo "Launching Arc browser with remote debugging enabled on port 9222..."
echo "This will allow the CodeAct browser automation to connect to it."

# Check if Arc is installed
if [ ! -d "/Applications/Arc.app" ]; then
  echo "Error: Arc browser not found in Applications folder."
  echo "If it's installed elsewhere, modify this script accordingly."
  exit 1
fi

# Launch Arc with remote debugging enabled
open -a "Arc" --args --remote-debugging-port=9222

echo "Arc browser launched with debugging enabled."
echo "In the CodeAct app, check 'Connect to existing browser' and use endpoint: http://localhost:9222" 