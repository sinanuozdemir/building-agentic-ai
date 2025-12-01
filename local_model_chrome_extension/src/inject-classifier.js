// inject-classifier.js - Handles classification in the context of web pages
// DO NOT import transformers.js here - we will load it dynamically

// Flag to avoid double initialization
window.modelInitialized = window.modelInitialized || false;
window.sentimentModel = window.sentimentModel || null;

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action !== 'classify-text') return;
    
    // Immediately acknowledge receipt of the message
    sendResponse({ status: 'received' });
    
    // Start the classification process
    classifyText(message.text);
    
    // Return true to indicate we handled the message
    return true;
});

// Classify text and show results
async function classifyText(text) {
    try {
        // Show loading indicator
        const loadingTooltip = document.createElement('div');
        loadingTooltip.id = 'sentiment-loading';
        Object.assign(loadingTooltip.style, {
            position: 'absolute',
            zIndex: '10000',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '14px',
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
            maxWidth: '300px'
        });
        loadingTooltip.textContent = 'Analyzing text...';
        
        // Position tooltip near the selected text
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            loadingTooltip.style.left = `${rect.left + window.scrollX}px`;
            loadingTooltip.style.top = `${rect.bottom + window.scrollY + 10}px`;
        }
        document.body.appendChild(loadingTooltip);

        // Check model status and classify the text using the background script
        chrome.runtime.sendMessage({ action: 'getModelStatus' }, (modelStatusResponse) => {
            if (modelStatusResponse.status !== 'loaded') {
                // Update loading message with progress if available
                if (modelStatusResponse.status === 'loading' && modelStatusResponse.progress) {
                    loadingTooltip.textContent = `Loading model: ${modelStatusResponse.progress}%`;
                } else {
                    loadingTooltip.textContent = 'Loading model...';
                }
                
                // Poll the status until model is ready
                const checkInterval = setInterval(() => {
                    chrome.runtime.sendMessage({ action: 'getModelStatus' }, (response) => {
                        if (response.status === 'loaded') {
                            clearInterval(checkInterval);
                            // Proceed with classification
                            classifyWithModel(text, loadingTooltip);
                        } else if (response.status === 'loading') {
                            loadingTooltip.textContent = `Loading model: ${response.progress}%`;
                        }
                    });
                }, 1000);
            } else {
                // Model is already loaded, proceed with classification
                classifyWithModel(text, loadingTooltip);
            }
        });
    } catch (error) {
        console.error('Classification error:', error);
        alert('Error classifying text: ' + error.message);
    }
}

// Helper function to classify text using the background script
function classifyWithModel(text, loadingTooltip) {
    chrome.runtime.sendMessage({ 
        action: 'classifyText', 
        text: text 
    }, (response) => {
        // Remove loading indicator
        if (loadingTooltip.parentNode) {
            document.body.removeChild(loadingTooltip);
        }
        
        if (response.error) {
            console.error('Classification error:', response.error);
            alert('Error classifying text: ' + response.error);
        } else {
            // Display result in a tooltip
            showTooltip(response.result);
        }
    });
}

// Function to show the tooltip with the classification result
function showTooltip(result) {
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.id = 'sentiment-tooltip';
    
    // Style the tooltip
    Object.assign(tooltip.style, {
        position: 'absolute',
        zIndex: '10000',
        background: result[0].label === 'POSITIVE' ? 'rgba(76, 175, 80, 0.9)' : 'rgba(244, 67, 54, 0.9)',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '4px',
        fontSize: '14px',
        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
        maxWidth: '300px',
        pointerEvents: 'none',
        transition: 'opacity 0.3s'
    });
    
    // Set tooltip content
    const label = result[0].label;
    const score = (result[0].score * 100).toFixed(1);
    tooltip.textContent = `${label} (${score}%)`;
    
    // Position tooltip near the selected text
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        tooltip.style.left = `${rect.left + window.scrollX}px`;
        tooltip.style.top = `${rect.bottom + window.scrollY + 10}px`;
    }
    
    // Add tooltip to page
    document.body.appendChild(tooltip);
    
    // Remove tooltip after 3 seconds
    setTimeout(() => {
        if (tooltip.parentNode) {
            tooltip.style.opacity = '0';
            setTimeout(() => {
                if (tooltip.parentNode) {
                    document.body.removeChild(tooltip);
                }
            }, 300);
        }
    }, 3000);
    
    // Also remove tooltip when clicking elsewhere
    document.addEventListener('click', function removeTooltip() {
        if (tooltip.parentNode) {
            document.body.removeChild(tooltip);
        }
        document.removeEventListener('click', removeTooltip);
    });
} 