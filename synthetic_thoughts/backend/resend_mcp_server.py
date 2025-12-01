from fastmcp import FastMCP, Context
import os
import httpx
import json
from typing import List, Optional, Dict, Any

# Create server instance
mcp = FastMCP(name="ResendEmailServer")

# You'll need to set your Resend API key as an environment variable
# export RESEND_API_KEY=your_api_key_here

@mcp.tool()
async def send_email(
    to: List[str],
    subject: str,
    html: str,
    from_address: Optional[str] = None,
    reply_to: Optional[List[str]] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    text: Optional[str] = None,
    tags: Optional[List[Dict[str, str]]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Sends an email using Resend API.
    
    Args:
        to: List of recipient email addresses
        subject: Email subject line
        html: HTML content of the email
        from_address: Sender email address (defaults to account default)
        reply_to: List of reply-to email addresses
        cc: List of CC recipients
        bcc: List of BCC recipients
        text: Plain text version of email
        tags: List of tags for categorizing emails
    
    Returns:
        Dictionary with email ID and status
    """
    api_key = os.environ.get("RESEND_API_KEY")
    
    # For testing purposes when no API key is available
    if not api_key or api_key == "test_key":
        if ctx:
            await ctx.info("Using test mode (no valid API key)")
            await ctx.info(f"Would send email to {', '.join(to)} with subject: {subject}")
        
        # Return mock response for testing
        return {
            "id": "test_email_123456",
            "from": from_address or "test@example.com",
            "to": to,
            "created_at": "2025-05-16T12:00:00.000Z",
            "status": "sent"
        }
    
    if ctx:
        await ctx.info(f"Sending email to {', '.join(to)} with subject: {subject}")
    
    payload = {
        "to": to,
        "subject": subject,
        "html": html
    }
    
    # Add optional parameters if provided
    if from_address:
        payload["from"] = from_address
    if reply_to:
        payload["reply_to"] = reply_to
    if cc:
        payload["cc"] = cc
    if bcc:
        payload["bcc"] = bcc
    if text:
        payload["text"] = text
    if tags:
        payload["tags"] = tags
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        
        if response.status_code >= 400:
            error_msg = f"Error sending email: {response.text}"
            if ctx:
                await ctx.error(error_msg)
            raise Exception(error_msg)
        
        result = response.json()
        if ctx:
            await ctx.info(f"Email sent successfully with ID: {result.get('id')}")
        return result

@mcp.resource("emails://{email_id}")
async def get_email_status(email_id: str) -> Dict[str, Any]:
    """
    Get the status of an email by its ID.
    
    Args:
        email_id: The ID of the email to check
        
    Returns:
        Dictionary with email details
    """
    api_key = os.environ.get("RESEND_API_KEY")
    
    # For testing purposes when no API key is available
    if not api_key or api_key == "test_key" or email_id.startswith("test_"):
        # Return mock response for testing
        return {
            "id": email_id,
            "object": "email",
            "to": ["recipient@example.com"],
            "from": "test@example.com",
            "created_at": "2025-05-16T12:00:00.000Z",
            "status": "delivered",
            "subject": "Test Email"
        }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.resend.com/emails/{email_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        if response.status_code >= 400:
            raise Exception(f"Error retrieving email: {response.text}")
        
        return response.json()

@mcp.prompt()
def confirmation_template(recipient_name: str, action: str) -> str:
    """
    Generate a confirmation email template.
    
    Args:
        recipient_name: Name of the recipient
        action: Action that was completed
        
    Returns:
        HTML string for a confirmation email
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #f5f5f5; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Confirmation</h2>
            </div>
            <div class="content">
                <p>Hello {recipient_name},</p>
                <p>We're confirming that {action} has been completed successfully.</p>
                <p>If you did not request this action, please contact support immediately.</p>
                <p>Thank you for using our service!</p>
            </div>
            <div class="footer">
                <p>This is an automated email, please do not reply directly.</p>
            </div>
        </div>
    </body>
    </html>
    """

@mcp.prompt()
def newsletter_template(recipient_name: str, main_content: str, cta_text: str, cta_link: str) -> str:
    """
    Generate a newsletter email template.
    
    Args:
        recipient_name: Name of the recipient
        main_content: Main content of the newsletter
        cta_text: Call to action button text
        cta_link: Call to action button link
        
    Returns:
        HTML string for a newsletter
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4CAF50; padding: 20px; text-align: center; color: white; }}
            .content {{ padding: 20px; }}
            .cta-button {{ display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; 
                          text-decoration: none; border-radius: 5px; margin: 20px 0; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Newsletter</h2>
            </div>
            <div class="content">
                <p>Hello {recipient_name},</p>
                {main_content}
                <p><a href="{cta_link}" class="cta-button">{cta_text}</a></p>
                <p>Thanks for subscribing to our newsletter!</p>
            </div>
            <div class="footer">
                <p>To unsubscribe from these emails, click <a href="#">here</a>.</p>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    # For streamable-http transport
    # mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
    
    # Default transport (STDIO)
    mcp.run() 