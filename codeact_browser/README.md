# Browser Automation Agent

A simple agent that controls a browser through a persistent service.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m playwright install
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Start the browser service:
```bash
./kill_server.sh
./run_service.sh
```

4. Run the Streamlit app:
```bash
streamlit run test_codeact.py
```

## Using Existing Browsers

You can now connect to an existing browser instead of having the agent create a new one:

1. Launch a browser with remote debugging enabled:
   - For Arc: `./launch_arc_debug.sh` or `open -a "Arc" --args --remote-debugging-port=9222`
   - For Chrome: `./launch_chrome_debug.sh` or `/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222`

2. In the Streamlit app sidebar:
   - Check "Connect to existing browser"
   - Use the default endpoint: `http://localhost:9222`
   - Click "New Conversation" to reconnect with your preferences

3. The agent will now control your existing browser instead of creating a new one.

### Benefits of using existing browsers:
- Access to your login sessions, bookmarks, and extensions
- Use the browser you're already familiar with
- Control a specific tab in a multi-tab browser session

## Example Commands

- "Go to example.com"
- "Search for 'Claude AI' on Google"
- "Scroll down"
- "Take a screenshot"
- "Click on the first result"
- "Go to wikipedia.org"
- "Type 'Artificial Intelligence' in the search box and submit"

## Troubleshooting

- If the browser isn't responding, check if the service is running
- The service translates "browser_page" to "page" automatically
- To restart the service: `./kill_server.sh && ./run_service.sh`
- For status information: "What is the current browser status?"
- When using an existing browser, make sure it was launched with the `--remote-debugging-port=9222` flag
- If connection fails, try restarting your browser with debugging enabled 