# Resend Email MCP Server

An MCP (Model Context Protocol) server for sending emails using the [Resend](https://resend.com) API, built with FastMCP.

## Features

- Send emails with HTML and text content
- Support for CC, BCC, reply-to, and custom tags
- Check email status
- Ready-to-use email templates for confirmations and newsletters
- Proper error handling and logging
- Async API support

## Installation

```bash
pip install fastmcp httpx
```

## Setup

1. Sign up for a [Resend](https://resend.com) account
2. Get your API key from the Resend dashboard
3. Set your API key as an environment variable:

```bash
export RESEND_API_KEY=your_api_key_here
```

## Usage

### Starting the server

```bash
python resend_mcp_server.py
```

By default, the server runs using the STDIO transport, making it suitable for local tool usage. 

To run as an HTTP server:

```python
# Modify the last lines in resend_mcp_server.py
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
```

### Available Tools

#### 1. Send Email

Send an email using the Resend API:

```python
await client.call_tool("send_email", {
    "to": ["recipient@example.com"],
    "subject": "Hello from MCP",
    "html": "<p>This is a test email</p>",
    "from_address": "sender@yourdomain.com"  # Optional
})
```

#### 2. Get Email Status

Check the status of a sent email:

```python
email_info = await client.read_resource("emails://email_id_here")
```

### Email Templates

#### Confirmation Template

```python
confirmation_html = await client.call_prompt("confirmation_template", {
    "recipient_name": "John Doe",
    "action": "your account registration"
})
```

#### Newsletter Template

```python
newsletter_html = await client.call_prompt("newsletter_template", {
    "recipient_name": "Jane Smith",
    "main_content": "<p>Check out our latest features...</p>",
    "cta_text": "Learn More",
    "cta_link": "https://example.com/features"
})
```

## Example: Sending a Newsletter

```python
from fastmcp import Client

async def send_newsletter():
    async with Client("./resend_mcp_server.py") as client:
        # Get newsletter template
        newsletter_html = await client.call_prompt("newsletter_template", {
            "recipient_name": "Subscriber",
            "main_content": "<p>We're excited to announce our new product launch!</p>",
            "cta_text": "View Product",
            "cta_link": "https://example.com/product"
        })
        
        # Send the email
        result = await client.call_tool("send_email", {
            "to": ["subscriber@example.com"],
            "subject": "New Product Announcement",
            "html": newsletter_html.text,
            "from_address": "news@yourdomain.com"
        })
        
        print(f"Email sent with ID: {result.text['id']}")

# Run the example
import asyncio
asyncio.run(send_newsletter())
```

## License

MIT
