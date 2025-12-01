// background.js - Handles model loading and context menu functionality
import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js to use local files instead of CDN
env.useBrowserCache = false;
env.allowLocalModels = true;
// Point to local WASM files
env.backends.onnx.wasm.wasmPaths = {
    'ort-wasm-simd-threaded.wasm': chrome.runtime.getURL('ort-wasm-simd-threaded.wasm'),
    'ort-wasm-simd.wasm': chrome.runtime.getURL('ort-wasm-simd.wasm'),
    'ort-wasm-threaded.wasm': chrome.runtime.getURL('ort-wasm-threaded.wasm'),
    'ort-wasm.wasm': chrome.runtime.getURL('ort-wasm.wasm')
};
// Point to local JS files
env.backends.onnx.wasm.jsEpPaths = {
    'ort-wasm-simd-threaded.jsep.mjs': chrome.runtime.getURL('ort-wasm-simd-threaded.jsep.mjs'),
    'ort-wasm-simd.jsep.mjs': chrome.runtime.getURL('ort-wasm-simd.jsep.mjs'),
    'ort-wasm-threaded.jsep.mjs': chrome.runtime.getURL('ort-wasm-threaded.jsep.mjs'),
    'ort-wasm.jsep.mjs': chrome.runtime.getURL('ort-wasm.jsep.mjs')
};

// Store the model globally
let sentimentModel = null;
let isModelLoading = false;
let modelLoadingProgress = 0;

// Load the model once when the extension starts
async function loadModel() {
    if (sentimentModel || isModelLoading) return;
    
    try {
        isModelLoading = true;
        // Use a simpler model that's less likely to have issues
        sentimentModel = await pipeline(
            'sentiment-analysis', 
            'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
            {
                quantized: false, // Disable quantization for more compatibility
                progress_callback: (progress) => {
                    if (progress.status === 'progress') {
                        modelLoadingProgress = Math.round(progress.progress);
                        console.log(`Model loading: ${modelLoadingProgress}%`);
                    }
                }
            }
        );
        console.log('Model loaded successfully');
        isModelLoading = false;
    } catch (error) {
        console.error('Error loading model:', error);
        isModelLoading = false;
    }
}

// Add a listener for messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'getModelStatus') {
        if (sentimentModel) {
            sendResponse({ status: 'loaded' });
        } else if (isModelLoading) {
            sendResponse({ status: 'loading', progress: modelLoadingProgress });
        } else {
            sendResponse({ status: 'not_loaded' });
            loadModel(); // Start loading if not already loading
        }
        return true; // Indicates we will send a response asynchronously
    } else if (message.action === 'classifyText') {
        if (!sentimentModel) {
            sendResponse({ error: 'Model not loaded yet' });
            return true;
        }
        
        // Classify the text
        sentimentModel(message.text)
            .then(result => {
                sendResponse({ result });
            })
            .catch(error => {
                sendResponse({ error: error.message });
            });
        
        return true; // Indicates we will send a response asynchronously
    }
});

////////////////////// Context Menus //////////////////////
// Add a listener to create the initial context menu items
chrome.runtime.onInstalled.addListener(function () {
    // Register a context menu item that will only show up for selection text
    chrome.contextMenus.create({
        id: 'classify-selection',
        title: 'Classify "%s"',
        contexts: ['selection'],
    });
    
    // Start loading the model when extension is installed/updated
    loadModel();
});

// Perform inference when the user clicks a context menu
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    // Ignore context menu clicks that are not for classifications
    if (info.menuItemId !== 'classify-selection' || !info.selectionText) return;

    try {
        // Run a function in the context of the page that will handle classification
        await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ['inject-classifier.js']
        });
        
        // After the script is loaded, send the message
        chrome.tabs.sendMessage(tab.id, {
            action: 'classify-text',
            text: info.selectionText
        });
    } catch (error) {
        console.error('Error executing script:', error);
    }
});

