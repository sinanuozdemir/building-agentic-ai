// popup.js - Use model from background script

const inputElement = document.getElementById('text');
const outputElement = document.getElementById('output');

// Initialize status message
outputElement.innerText = 'Checking model status...';

// Check the model status from the background script
async function checkModelStatus() {
    chrome.runtime.sendMessage({ action: 'getModelStatus' }, (response) => {
        if (response.status === 'loaded') {
            outputElement.innerText = 'Model loaded! Type text to analyze.';
        } else if (response.status === 'loading') {
            outputElement.innerText = `Loading model: ${response.progress}%`;
            // Check again in 1 second
            setTimeout(checkModelStatus, 1000);
        } else {
            outputElement.innerText = 'Starting model loading...';
            // Check again in 1 second
            setTimeout(checkModelStatus, 1000);
        }
    });
}

// Start checking status
checkModelStatus();

// Listen for input
inputElement.addEventListener('input', async (event) => {
    const text = event.target.value;
    
    if (!text || text.trim().length === 0) {
        outputElement.innerText = 'Enter text to analyze sentiment';
        return;
    }
    
    // Check model status first
    chrome.runtime.sendMessage({ action: 'getModelStatus' }, (modelStatusResponse) => {
        if (modelStatusResponse.status !== 'loaded') {
            outputElement.innerText = 'Model is still loading...';
            return;
        }
        
        // If model is loaded, send text for classification
        outputElement.innerText = 'Analyzing...';
        chrome.runtime.sendMessage({ 
            action: 'classifyText', 
            text: text 
        }, (response) => {
            if (response.error) {
                outputElement.innerText = `Error analyzing text: ${response.error}`;
            } else {
                outputElement.innerText = JSON.stringify(response.result, null, 2);
            }
        });
    });
});
