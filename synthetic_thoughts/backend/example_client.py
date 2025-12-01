from fastmcp import Client
import asyncio
import os
import json

async def main():
    # Connect to the Resend MCP server
    async with Client("./resend_mcp_server.py") as client:
        print("Connected to Resend MCP server")
        
        # List all available tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)} found")
        for tool in tools:
            print(f"- {tool.name}")
        
        # List all available prompts
        prompts = await client.list_prompts()
        print(f"Available prompts: {len(prompts)} found")
        for prompt in prompts:
            print(f"- {prompt.name}")
        
        # Example: Send a simple email
        print("\n--- Sending a test email ---")
        try:
            result = await client.call_tool("send_email", {
                "to": ["recipient@example.com"],
                "subject": "Test from MCP Client",
                "html": "<h1>Hello from MCP!</h1><p>This is a test email from FastMCP.</p>"
            })
            
            # Handle the response properly (it may be different in this FastMCP version)
            print(f"Email sent successfully: {result}")
            
            if isinstance(result, dict) and "id" in result:
                email_id = result["id"]
            elif hasattr(result, "text") and isinstance(result.text, dict) and "id" in result.text:
                email_id = result.text["id"]
            elif isinstance(result, list) and len(result) > 0:
                # Try to extract the first item if it's a list
                first_item = result[0]
                if isinstance(first_item, dict) and "id" in first_item:
                    email_id = first_item["id"]
                else:
                    email_id = "test_email_123456"  # Fallback ID
            else:
                email_id = "test_email_123456"  # Fallback ID
            
            # Check email status
            print(f"\nChecking email status for ID: {email_id}...")
            status = await client.read_resource(f"emails://{email_id}")
            print(f"Email status: {status}")
        except Exception as e:
            print(f"Error sending email: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Set a test key for demo purposes
    os.environ["RESEND_API_KEY"] = "test_key"
    asyncio.run(main()) 