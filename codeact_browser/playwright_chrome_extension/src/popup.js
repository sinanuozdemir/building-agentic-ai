document.addEventListener('DOMContentLoaded', function() {
  // Log that popup is loaded
  console.log("[POPUP] Loaded at:", new Date().toISOString());
  
  const codeInput = document.getElementById('code-input');
  const executeButton = document.getElementById('execute');
  const loadSampleButton = document.getElementById('load-sample');
  const statusElement = document.getElementById('status');
  const consoleOutput = document.getElementById('console-output');
  const resultOutput = document.getElementById('result-output');
  const tabs = document.querySelectorAll('.tab');
  const tabContents = document.querySelectorAll('.tab-content');
  const historyOutput = document.getElementById('history-output');
  
  // Pusher elements
  const pusherAppKeyInput = document.getElementById('pusher-app-key');
  const pusherClusterInput = document.getElementById('pusher-cluster');
  const pusherChannelInput = document.getElementById('pusher-channel');
  const pusherEventInput = document.getElementById('pusher-event');
  const pusherConnectButton = document.getElementById('pusher-connect');
  const pusherDisconnectButton = document.getElementById('pusher-disconnect');
  const pusherStatusElement = document.getElementById('pusher-status');

  console.log("[POPUP] DOM elements initialized");

  // Default Pusher configuration based on sample
  const defaultPusherConfig = {
    appKey: '271b88729ca02b9f059d',
    cluster: 'us3',
    channelName: 'my-channel',
    eventName: 'my-event'
  };

  // initial history render
  renderHistory();
  
  // Check Pusher connection status
  checkPusherStatus();

  // Sample Playwright-like code
  const sampleCode = `// Go to my site
await page.goto('https://sinanozdemir.ai');

// Click the first link that appears
await page.waitForSelector('a');
await page.click('a');

// Wait for navigation to complete and ensure title is ready
await page.waitForSelector('title');

return {
  finalURL: window.location.href,
  finalTitle: document.title,
};`;

  // Set up tab switching
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      console.log("[POPUP] Tab clicked:", tab.getAttribute('data-tab'));
      // Deactivate all tabs
      tabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      // Activate the clicked tab
      tab.classList.add('active');
      const tabContent = document.getElementById(tab.getAttribute('data-tab'));
      tabContent.classList.add('active');
    });
  });

  // Load previously saved code (if any)
  chrome.storage.local.get(['playwrightCode', 'pusherConfig'], function(result) {
    console.log("[POPUP] Storage retrieved:", result);
    if (result.playwrightCode) {
      codeInput.value = result.playwrightCode;
    }
    
    // Load saved Pusher config or use defaults
    const config = result.pusherConfig || defaultPusherConfig;
    pusherAppKeyInput.value = config.appKey || defaultPusherConfig.appKey;
    pusherClusterInput.value = config.cluster || defaultPusherConfig.cluster;
    pusherChannelInput.value = config.channelName || defaultPusherConfig.channelName;
    pusherEventInput.value = config.eventName || defaultPusherConfig.eventName;
  });

  // Load sample code when the button is clicked
  loadSampleButton.addEventListener('click', function() {
    console.log("[POPUP] Load sample button clicked");
    codeInput.value = sampleCode;
    updateStatus('Sample code loaded! Click "Execute Code" to run it.', 'success');
  });

  executeButton.addEventListener('click', async function() {
    console.log("[POPUP] Execute button clicked");
    const code = codeInput.value.trim();
    
    if (!code) {
      updateStatus('Please enter some Playwright code', 'error');
      return;
    }

    // Clear previous output
    consoleOutput.innerHTML = '';
    resultOutput.innerHTML = '';
    
    // Switch to output tab
    tabs[1].click();

    // Save the code for next time
    chrome.storage.local.set({ playwrightCode: code });
    console.log("[POPUP] Code saved to storage");
    
    try {
      updateStatus('Executing code...', 'pending');
      
      // Get the active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      console.log("[POPUP] Executing on tab:", tab.id, tab.url);
      
      // Send the code to the background script for execution
      chrome.runtime.sendMessage({ 
        action: 'executePlaywrightCode', 
        code: code,
        tabId: tab.id 
      }, function(response) {
        console.log("[POPUP] Received response from background:", response);
        
        if (response && response.success && response.result !== undefined) {
          updateStatus('Code executed successfully!', 'success');
          
          // Display console logs
          if (response.logs && response.logs.length > 0) {
            console.log("[POPUP] Displaying logs:", response.logs);
            response.logs.forEach(logEntry => {
              const [type, message] = logEntry;
              const logElement = document.createElement('div');
              logElement.className = `console-${type}`;
              
              // Check if this is a line-specific error message
              const lineErrorMatch = message.match(/^\[Line (\d+)\] (.+)/);
              if (lineErrorMatch && type === 'error') {
                const lineNum = lineErrorMatch[1];
                const errorMsg = lineErrorMatch[2];
                logElement.innerHTML = `<span class="line-number">Line ${lineNum}:</span> <span class="error-message">${errorMsg}</span>`;
                logElement.style.fontWeight = 'bold';
              } else {
                logElement.textContent = message;
              }
              
              consoleOutput.appendChild(logElement);
            });
          } else {
            console.log("[POPUP] No logs to display");
            consoleOutput.innerHTML = '<div class="console-log">No console output</div>';
          }
          
          // Display the result
          if (response.result !== undefined) {
            console.log("[POPUP] Displaying result:", response.result);
            try {
              const resultStr = JSON.stringify(response.result, null, 2);
              resultOutput.innerHTML = '<pre>' + resultStr + '</pre>';
            } catch (e) {
              resultOutput.innerHTML = '<pre>Error: Could not format result</pre>';
            }
          } else {
            console.log("[POPUP] No result to display");
            resultOutput.innerHTML = '<div>No return value</div>';
          }

          // Show page info if available
          if (response.pageInfo) {
            console.log("[POPUP] Displaying page info:", response.pageInfo);
            const pageInfo = document.createElement('div');
            pageInfo.className = 'console-log';
            pageInfo.innerHTML = '<strong>Page Info:</strong><br>' +
              'Title: ' + response.pageInfo.title + '<br>' +
              'URL: ' + response.pageInfo.url;
            consoleOutput.appendChild(pageInfo);
          }

          // Refresh history list
          renderHistory();
        } else {
          console.log("[POPUP] Error response:", response);
          updateStatus(`Error: ${response ? response.error : 'No response from execution'}`, 'error');
          
          // Fallback: try to retrieve lastExecution from storage (in case response was lost)
          chrome.storage.local.get(['lastExecution'], (res) => {
            if (res && res.lastExecution && res.lastExecution.result !== undefined) {
              console.log('[POPUP] Fallback to lastExecution:', res.lastExecution);
              updateStatus('Recovered execution result.', 'success');
              const { result: recoveredResult, logs: recoveredLogs } = res.lastExecution;

              if (recoveredLogs && recoveredLogs.length) {
                recoveredLogs.forEach(logEntry => {
                  const [type, message] = logEntry;
                  const logElement = document.createElement('div');
                  logElement.className = `console-${type}`;
                  logElement.textContent = message;
                  consoleOutput.appendChild(logElement);
                });
              }

              if (recoveredResult !== undefined) {
                try {
                  const resultStr = JSON.stringify(recoveredResult, null, 2);
                  resultOutput.innerHTML = '<pre>' + resultStr + '</pre>';
                } catch (e) {
                  resultOutput.innerHTML = '<pre>Error: Could not format result</pre>';
                }
              }

              // Refresh history list even on fallback success
              renderHistory();
            } else {
              const errorElement = document.createElement('div');
              errorElement.className = 'console-error';
              errorElement.textContent = response ? (response.error || 'No response from execution') : 'No response from execution';
              consoleOutput.appendChild(errorElement);
            }
          });
        }
      });
    } catch (error) {
      console.error("[POPUP] Error:", error);
      updateStatus(`Error: ${error.message}`, 'error');
      
      // Display error in console output
      const errorElement = document.createElement('div');
      errorElement.className = 'console-error';
      errorElement.textContent = error.message;
      consoleOutput.appendChild(errorElement);
    }
  });
  
  // Pusher connect button click handler
  pusherConnectButton.addEventListener('click', function() {
    const appKey = pusherAppKeyInput.value.trim();
    const cluster = pusherClusterInput.value.trim();
    const channelName = pusherChannelInput.value.trim();
    const eventName = pusherEventInput.value.trim();
    
    if (!appKey || !channelName || !eventName) {
      updatePusherStatus(false, 'Missing required Pusher configuration');
      return;
    }
    
    const config = {
      appKey,
      cluster,
      channelName,
      eventName
    };
    
    // Save the config for next time
    chrome.storage.local.set({ pusherConfig: config });
    
    // Send the config to the background script to establish connection
    chrome.runtime.sendMessage({
      action: 'connectPusher',
      config
    }, function(response) {
      console.log("[POPUP] Connect Pusher response:", response);
      if (response && response.success) {
        updatePusherStatus(true, 'Connected to Pusher');
        pusherConnectButton.disabled = true;
        pusherDisconnectButton.disabled = false;
      } else {
        updatePusherStatus(false, 'Failed to connect to Pusher');
      }
    });
  });
  
  // Pusher disconnect button click handler
  pusherDisconnectButton.addEventListener('click', function() {
    chrome.runtime.sendMessage({
      action: 'disconnectPusher'
    }, function(response) {
      console.log("[POPUP] Disconnect Pusher response:", response);
      if (response && response.success) {
        updatePusherStatus(false, 'Disconnected from Pusher');
        pusherConnectButton.disabled = false;
        pusherDisconnectButton.disabled = true;
      }
    });
  });
  
  // Check Pusher connection status
  function checkPusherStatus() {
    chrome.runtime.sendMessage({
      action: 'checkPusherStatus'
    }, function(response) {
      console.log("[POPUP] Check Pusher status response:", response);
      if (response) {
        updatePusherStatus(response.connected, response.connected ? 'Connected to Pusher' : 'Disconnected from Pusher');
        
        if (response.connected) {
          pusherConnectButton.disabled = true;
          pusherDisconnectButton.disabled = false;
          
          // Load the saved config into the form
          if (response.config) {
            pusherAppKeyInput.value = response.config.appKey || '';
            pusherClusterInput.value = response.config.cluster || 'us2';
            pusherChannelInput.value = response.config.channelName || '';
            pusherEventInput.value = response.config.eventName || 'playwright-cmd';
          }
        } else {
          pusherConnectButton.disabled = false;
          pusherDisconnectButton.disabled = true;
        }
      }
    });
  }
  
  // Update Pusher connection status UI
  function updatePusherStatus(connected, message) {
    const statusIcon = pusherStatusElement.querySelector('.connection-status');
    const statusText = pusherStatusElement.querySelector('span:last-child');
    
    if (connected) {
      statusIcon.classList.remove('disconnected');
      statusIcon.classList.add('connected');
    } else {
      statusIcon.classList.remove('connected');
      statusIcon.classList.add('disconnected');
    }
    
    statusText.textContent = message || (connected ? 'Connected' : 'Disconnected');
  }

  function updateStatus(message, type) {
    console.log("[POPUP] Status update:", message, type);
    statusElement.textContent = message;
    statusElement.className = type || '';
  }
  
  function renderHistory() {
    chrome.storage.local.get(['history'], (data) => {
      const history = Array.isArray(data.history) ? data.history : [];
      historyOutput.innerHTML = '';
      if (!history.length) {
        historyOutput.textContent = 'No runs yet.';
        return;
      }
      history.slice().reverse().forEach(rec => {
        const div = document.createElement('div');
        const date = new Date(rec.ts).toLocaleTimeString();
        div.innerHTML = `<strong>${date}</strong>: <code>${JSON.stringify(rec.result)}</code>`;
        div.style.marginBottom = '5px';
        historyOutput.appendChild(div);
      });
    });
  }
  
  console.log("[POPUP] Initialization complete");
}); 