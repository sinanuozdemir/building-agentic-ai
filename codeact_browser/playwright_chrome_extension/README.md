# Playwright Code Executor Chrome Extension

This Chrome extension allows you to execute Playwright-like JavaScript code directly on the current tab.

## Features

- Simple popup interface to input Playwright code
- Execute JavaScript code that uses a simplified Playwright-like API
- Perform common browser automation tasks like:
  - Navigation (`page.goto()`)
  - Clicks (`page.click()`)
  - Form filling (`page.fill()`)
  - Content extraction (`page.textContent()`)
  - Waiting for elements (`page.waitForSelector()`)
  - Running arbitrary JavaScript (`page.evaluate()`)

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" by toggling the switch in the top right corner
3. Click "Load unpacked" and select the `playwright_chrome_extension` directory
4. The extension should now be installed and ready to use

## Usage

1. Click the extension icon in Chrome toolbar
2. Enter your Playwright code in the textarea
3. Click "Execute Code" to run it on the current tab

## Example Code

```javascript
// Navigate to a website
await page.goto('https://example.com');

// Wait for an element to appear and click it
await page.waitForSelector('.button');
await page.click('.button');

// Fill a form field
await page.fill('#username', 'testuser');

// Extract text content
const text = await page.textContent('.result');
console.log(text);
```

## Limitations

- This is a simplified version of Playwright - not all API methods are supported
- Code runs in the context of the browser tab, not in a separate browser instance
- Security restrictions apply as per Chrome extension policies

## Troubleshooting

If your code doesn't execute properly, check:
- Syntax errors in your code
- Selectors that might not exist on the page
- Browser restrictions that might prevent certain operations 