// This script is injected into all pages
// It helps to establish communication between the page and the extension

// Listen for messages from the background script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "getPageInfo") {
    // Gather page information that might be useful for Playwright operations
    const pageInfo = {
      url: window.location.href,
      title: document.title,
      readyState: document.readyState
    };
    sendResponse(pageInfo);
  }
  
  return true; // Keep the message channel open for async responses
}); 