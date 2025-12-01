# """
# This script creates an updated CodeAct wrapper that maps 'browser_page' to 'page' for the browser service.
# """

# from browser_client import BrowserServiceClient
# import time
# import re
# import sys
# import requests

# # Create browser service client
# browser_client = BrowserServiceClient("http://localhost:8000")

# def fix_browser_variable_names(code):
#     """Replace browser_page with page in the code, to match what the service expects."""
#     # Replace browser_page with page
#     fixed_code = code.replace("browser_page.", "page.")
#     fixed_code = fixed_code.replace("browser_page =", "page =")
#     return fixed_code

# def eval_code_with_fixed_names(code, _locals):
#     """Execute code with the browser service, fixing variable names first."""
#     session_id = _locals.get("session_id", "default")
#     print(f"Using session ID: {session_id}")
    
#     # Check browser service health directly
#     try:
#         response = requests.get(f"http://localhost:8000/")
#         print(f"Service health check: {response.status_code} - {response.text}")
#     except Exception as e:
#         print(f"Error checking service health: {str(e)}")
    
#     # Check browser status
#     success, status = browser_client.get_browser_status(session_id)
#     print(f"Browser status check: {success}, {status}")
    
#     if not success:
#         print(f"Browser status check failed: {status}", file=sys.stderr)
#         return f"⚠️ Error: Browser is not available. {status}", {}
    
#     # Fix variable names in the code
#     fixed_code = fix_browser_variable_names(code)
    
#     # Print the transformation for debugging
#     print("\nOriginal code:")
#     print(code)
#     print("\nFixed code:")
#     print(fixed_code)
    
#     # Execute the fixed code
#     print(f"Executing code with session_id: {session_id}")
#     success, output, screenshot_path = browser_client.execute_code(
#         fixed_code, session_id
#     )
    
#     print(f"Execution result: {success}, {output}")
    
#     if success:
#         result = output
#         if screenshot_path:
#             result += f"\n[Screenshot saved to {screenshot_path}]"
#     else:
#         result = f"⚠️ Error during execution: {output}"
    
#     # Return the result with an empty vars dict since browser service doesn't pass back vars
#     return result, {}

# def main():
#     """Test the fixed code execution."""
#     # Check if browser service is running
#     if not browser_client.health_check():
#         print("Browser service is not running. Please start it with ./run_service.sh")
#         return
    
#     # Initialize browser with a unique session ID
#     session_id = str(time.time())
#     print(f"Initializing browser with session ID: {session_id}")
    
#     success, message = browser_client.init_browser(session_id)
#     print(f"Browser initialization result: {success}, {message}")
    
#     if not success:
#         print(f"Failed to initialize browser: {message}")
#         return
    
#     print(f"Browser initialized with session ID: {session_id}")
    
#     # Test direct API endpoint
#     try:
#         response = requests.get(f"http://localhost:8000/browser/status?session_id={session_id}")
#         print(f"Direct status check: {response.status_code} - {response.text}")
#     except Exception as e:
#         print(f"Error with direct status check: {str(e)}")
    
#     # Test both variable naming conventions
#     test_codes = [
#         # Original naming that needs fixing
#         """
#         # Test with browser_page (needs fixing)
#         browser_page.goto('https://sinanozdemir.ai')
#         title = browser_page.title()
#         output = f"Page title: {title}"
#         """,
        
#         # Already correct naming
#         """
#         # Test with page (already correct)
#         page.goto('https://google.com')
#         title = page.title()
#         output = f"Page title: {title}"
#         """
#     ]
    
#     # Run tests
#     for code in test_codes:
#         print("\n=== Testing code ===")
#         output, _ = eval_code_with_fixed_names(code, {"session_id": session_id})
#         print(f"Result: {output}")
    
#     # Clean up
#     print(f"Closing browser for session {session_id}")
#     success, message = browser_client.close_browser(session_id)
#     print(f"Browser closed: {success}, {message}")

# if __name__ == "__main__":
#     main() 