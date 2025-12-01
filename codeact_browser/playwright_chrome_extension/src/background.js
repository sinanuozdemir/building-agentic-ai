// Background script for Playwright Code Executor using chrome.debugger API (step-by-step driver)

// Import config file for secrets
importScripts('config.js');

// Remove Pusher import as it doesn't work in service workers
// importScripts('pusher.min.js');

// Helpful startup log so we know the service-worker actually loaded
console.log('[PW-Exec] Background service-worker initialised', new Date().toISOString());

// Keep a ping to show when the worker is about to go idle (helps during debugging)
self.addEventListener('onstatechange', (e) => {
  console.log('[PW-Exec] Worker state change:', self.state);
});

// Custom WebSocket connection for Pusher
let ws = null;
let wsReconnectTimer = null;
let pusherConfig = {
  appKey: '',
  cluster: '',
  channelName: '',
  eventName: ''
};

/**
 * Send result back to Pusher when responseChannel and responseEvent are provided
 */
async function sendResultToPusher(channel, event, data, appKey, cluster) {
  try {
    console.log('[PW-Exec] Preparing to send results to Pusher:', { channel, event, data });
    
    // Format the data for Pusher
    const requestBody = {
      name: event,
      channel: channel,
      data: JSON.stringify(data)  // Pusher requires data to be a string
    };

    // Generate timestamp for auth
    const timestamp = Math.floor(Date.now() / 1000);
    
    // Calculate MD5 hash of the request body
    const bodyString = JSON.stringify(requestBody);
    const bodyMD5 = calculateMD5(bodyString);  // Now a synchronous call
    
    // For debugging only - log what we're sending
    console.log('[PW-Exec] Request body for Pusher:', bodyString);
    console.log('[PW-Exec] MD5 hash:', bodyMD5);
    
    // Authentication parameters
    const authParams = {
      auth_key: appKey,
      auth_timestamp: timestamp,
      auth_version: "1.0",
      body_md5: bodyMD5
    };
    
    // Create the query string to sign
    const queryString = Object.entries(authParams)
      .map(([key, value]) => `${key}=${value}`)
      .sort()
      .join('&');
    
    // Create the string to sign
    const stringToSign = `POST\n/apps/${PUSHER_CONFIG.APP_ID}/events\n${queryString}`;
    
    // Generate the signature
    const signature = await calculateHMACSHA256(stringToSign, PUSHER_CONFIG.APP_SECRET);
    
    // Full URL with authentication parameters
    const url = `https://api-${cluster}.pusher.com/apps/${PUSHER_CONFIG.APP_ID}/events?${queryString}&auth_signature=${signature}`;
    
    // Make the request to Pusher API
    console.log('[PW-Exec] Sending results to Pusher channel:', channel, 'event:', event);
    console.log('[PW-Exec] Pusher API URL:', url);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: bodyString
    });
    
    if (response.ok) {
      console.log('[PW-Exec] Results sent successfully to Pusher!');
      const responseText = await response.text();
      console.log('[PW-Exec] Pusher response:', responseText);
    } else {
      const errorText = await response.text();
      console.error('[PW-Exec] Failed to send results to Pusher:', response.status, errorText);
    }
  } catch (error) {
    console.error('[PW-Exec] Error sending results to Pusher:', error.message);
    console.error('[PW-Exec] Error details:', error);
  }
}

// Helper function to calculate MD5 hash
function calculateMD5(string) {
  // A simpler MD5 implementation
  function md5(string) {
    function rotateLeft(lValue, iShiftBits) {
      return (lValue << iShiftBits) | (lValue >>> (32 - iShiftBits));
    }

    function addUnsigned(lX, lY) {
      const lX8 = (lX & 0x80000000);
      const lY8 = (lY & 0x80000000);
      const lX4 = (lX & 0x40000000);
      const lY4 = (lY & 0x40000000);
      const lResult = (lX & 0x3FFFFFFF) + (lY & 0x3FFFFFFF);
      if (lX4 & lY4) {
        return (lResult ^ 0x80000000 ^ lX8 ^ lY8);
      }
      if (lX4 | lY4) {
        if (lResult & 0x40000000) {
          return (lResult ^ 0xC0000000 ^ lX8 ^ lY8);
        } else {
          return (lResult ^ 0x40000000 ^ lX8 ^ lY8);
        }
      } else {
        return (lResult ^ lX8 ^ lY8);
      }
    }

    function F(x, y, z) { return (x & y) | ((~x) & z); }
    function G(x, y, z) { return (x & z) | (y & (~z)); }
    function H(x, y, z) { return (x ^ y ^ z); }
    function I(x, y, z) { return (y ^ (x | (~z))); }

    function FF(a, b, c, d, x, s, ac) {
      a = addUnsigned(a, addUnsigned(addUnsigned(F(b, c, d), x), ac));
      return addUnsigned(rotateLeft(a, s), b);
    }

    function GG(a, b, c, d, x, s, ac) {
      a = addUnsigned(a, addUnsigned(addUnsigned(G(b, c, d), x), ac));
      return addUnsigned(rotateLeft(a, s), b);
    }

    function HH(a, b, c, d, x, s, ac) {
      a = addUnsigned(a, addUnsigned(addUnsigned(H(b, c, d), x), ac));
      return addUnsigned(rotateLeft(a, s), b);
    }

    function II(a, b, c, d, x, s, ac) {
      a = addUnsigned(a, addUnsigned(addUnsigned(I(b, c, d), x), ac));
      return addUnsigned(rotateLeft(a, s), b);
    }

    function convertToWordArray(string) {
      let lWordCount;
      const lMessageLength = string.length;
      const lNumberOfWordsTemp1 = lMessageLength + 8;
      const lNumberOfWordsTemp2 = (lNumberOfWordsTemp1 - (lNumberOfWordsTemp1 % 64)) / 64;
      const lNumberOfWords = (lNumberOfWordsTemp2 + 1) * 16;
      const lWordArray = Array(lNumberOfWords - 1);
      let lBytePosition = 0;
      let lByteCount = 0;
      while (lByteCount < lMessageLength) {
        lWordCount = (lByteCount - (lByteCount % 4)) / 4;
        lBytePosition = (lByteCount % 4) * 8;
        lWordArray[lWordCount] = (lWordArray[lWordCount] | (string.charCodeAt(lByteCount) << lBytePosition));
        lByteCount++;
      }
      lWordCount = (lByteCount - (lByteCount % 4)) / 4;
      lBytePosition = (lByteCount % 4) * 8;
      lWordArray[lWordCount] = lWordArray[lWordCount] | (0x80 << lBytePosition);
      lWordArray[lNumberOfWords - 2] = lMessageLength << 3;
      lWordArray[lNumberOfWords - 1] = lMessageLength >>> 29;
      return lWordArray;
    }

    function wordToHex(lValue) {
      let wordToHexValue = "", wordToHexValueTemp = "", lByte, lCount;
      for (lCount = 0; lCount <= 3; lCount++) {
        lByte = (lValue >>> (lCount * 8)) & 255;
        wordToHexValueTemp = "0" + lByte.toString(16);
        wordToHexValue = wordToHexValue + wordToHexValueTemp.substr(wordToHexValueTemp.length - 2, 2);
      }
      return wordToHexValue;
    }

    let x = [];
    let k, AA, BB, CC, DD, a, b, c, d;
    const S11 = 7, S12 = 12, S13 = 17, S14 = 22;
    const S21 = 5, S22 = 9, S23 = 14, S24 = 20;
    const S31 = 4, S32 = 11, S33 = 16, S34 = 23;
    const S41 = 6, S42 = 10, S43 = 15, S44 = 21;

    // UTF8 encode
    string = unescape(encodeURIComponent(string));
    
    x = convertToWordArray(string);
    a = 0x67452301; b = 0xEFCDAB89; c = 0x98BADCFE; d = 0x10325476;

    for (k = 0; k < x.length; k += 16) {
      AA = a; BB = b; CC = c; DD = d;
      a = FF(a, b, c, d, x[k+0], S11, 0xD76AA478);
      d = FF(d, a, b, c, x[k+1], S12, 0xE8C7B756);
      c = FF(c, d, a, b, x[k+2], S13, 0x242070DB);
      b = FF(b, c, d, a, x[k+3], S14, 0xC1BDCEEE);
      a = FF(a, b, c, d, x[k+4], S11, 0xF57C0FAF);
      d = FF(d, a, b, c, x[k+5], S12, 0x4787C62A);
      c = FF(c, d, a, b, x[k+6], S13, 0xA8304613);
      b = FF(b, c, d, a, x[k+7], S14, 0xFD469501);
      a = FF(a, b, c, d, x[k+8], S11, 0x698098D8);
      d = FF(d, a, b, c, x[k+9], S12, 0x8B44F7AF);
      c = FF(c, d, a, b, x[k+10], S13, 0xFFFF5BB1);
      b = FF(b, c, d, a, x[k+11], S14, 0x895CD7BE);
      a = FF(a, b, c, d, x[k+12], S11, 0x6B901122);
      d = FF(d, a, b, c, x[k+13], S12, 0xFD987193);
      c = FF(c, d, a, b, x[k+14], S13, 0xA679438E);
      b = FF(b, c, d, a, x[k+15], S14, 0x49B40821);
      a = GG(a, b, c, d, x[k+1], S21, 0xF61E2562);
      d = GG(d, a, b, c, x[k+6], S22, 0xC040B340);
      c = GG(c, d, a, b, x[k+11], S23, 0x265E5A51);
      b = GG(b, c, d, a, x[k+0], S24, 0xE9B6C7AA);
      a = GG(a, b, c, d, x[k+5], S21, 0xD62F105D);
      d = GG(d, a, b, c, x[k+10], S22, 0x2441453);
      c = GG(c, d, a, b, x[k+15], S23, 0xD8A1E681);
      b = GG(b, c, d, a, x[k+4], S24, 0xE7D3FBC8);
      a = GG(a, b, c, d, x[k+9], S21, 0x21E1CDE6);
      d = GG(d, a, b, c, x[k+14], S22, 0xC33707D6);
      c = GG(c, d, a, b, x[k+3], S23, 0xF4D50D87);
      b = GG(b, c, d, a, x[k+8], S24, 0x455A14ED);
      a = GG(a, b, c, d, x[k+13], S21, 0xA9E3E905);
      d = GG(d, a, b, c, x[k+2], S22, 0xFCEFA3F8);
      c = GG(c, d, a, b, x[k+7], S23, 0x676F02D9);
      b = GG(b, c, d, a, x[k+12], S24, 0x8D2A4C8A);
      a = HH(a, b, c, d, x[k+5], S31, 0xFFFA3942);
      d = HH(d, a, b, c, x[k+8], S32, 0x8771F681);
      c = HH(c, d, a, b, x[k+11], S33, 0x6D9D6122);
      b = HH(b, c, d, a, x[k+14], S34, 0xFDE5380C);
      a = HH(a, b, c, d, x[k+1], S31, 0xA4BEEA44);
      d = HH(d, a, b, c, x[k+4], S32, 0x4BDECFA9);
      c = HH(c, d, a, b, x[k+7], S33, 0xF6BB4B60);
      b = HH(b, c, d, a, x[k+10], S34, 0xBEBFBC70);
      a = HH(a, b, c, d, x[k+13], S31, 0x289B7EC6);
      d = HH(d, a, b, c, x[k+0], S32, 0xEAA127FA);
      c = HH(c, d, a, b, x[k+3], S33, 0xD4EF3085);
      b = HH(b, c, d, a, x[k+6], S34, 0x4881D05);
      a = HH(a, b, c, d, x[k+9], S31, 0xD9D4D039);
      d = HH(d, a, b, c, x[k+12], S32, 0xE6DB99E5);
      c = HH(c, d, a, b, x[k+15], S33, 0x1FA27CF8);
      b = HH(b, c, d, a, x[k+2], S34, 0xC4AC5665);
      a = II(a, b, c, d, x[k+0], S41, 0xF4292244);
      d = II(d, a, b, c, x[k+7], S42, 0x432AFF97);
      c = II(c, d, a, b, x[k+14], S43, 0xAB9423A7);
      b = II(b, c, d, a, x[k+5], S44, 0xFC93A039);
      a = II(a, b, c, d, x[k+12], S41, 0x655B59C3);
      d = II(d, a, b, c, x[k+3], S42, 0x8F0CCC92);
      c = II(c, d, a, b, x[k+10], S43, 0xFFEFF47D);
      b = II(b, c, d, a, x[k+1], S44, 0x85845DD1);
      a = II(a, b, c, d, x[k+8], S41, 0x6FA87E4F);
      d = II(d, a, b, c, x[k+15], S42, 0xFE2CE6E0);
      c = II(c, d, a, b, x[k+6], S43, 0xA3014314);
      b = II(b, c, d, a, x[k+13], S44, 0x4E0811A1);
      a = II(a, b, c, d, x[k+4], S41, 0xF7537E82);
      d = II(d, a, b, c, x[k+11], S42, 0xBD3AF235);
      c = II(c, d, a, b, x[k+2], S43, 0x2AD7D2BB);
      b = II(b, c, d, a, x[k+9], S44, 0xEB86D391);
      a = addUnsigned(a, AA);
      b = addUnsigned(b, BB);
      c = addUnsigned(c, CC);
      d = addUnsigned(d, DD);
    }

    const result = wordToHex(a) + wordToHex(b) + wordToHex(c) + wordToHex(d);
    return result.toLowerCase();
  }

  return md5(string);
}

// Helper function to calculate HMAC SHA256
async function calculateHMACSHA256(message, secret) {
  // Convert the secret to a key
  const keyData = new TextEncoder().encode(secret);
  const key = await crypto.subtle.importKey(
    'raw', 
    keyData,
    { name: 'HMAC', hash: 'SHA-256' },
    false, 
    ['sign']
  );
  
  // Sign the message
  const msgUint8 = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.sign('HMAC', key, msgUint8);
  
  // Convert to hex string
  return Array.from(new Uint8Array(hashBuffer))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Initialize and connect to Pusher via WebSocket
 */
function connectToPusher(config) {
  if (!config.appKey || !config.channelName || !config.eventName) {
    console.error('[PW-Exec] Missing required Pusher configuration');
    return;
  }
  
  // Store the config for reconnection
  pusherConfig = config;
  
  // Disconnect existing connection if any
  disconnectPusher();
  
  try {
    // Initialize direct WebSocket connection to Pusher
    console.log('[PW-Exec] Connecting to Pusher with config:', pusherConfig);
    
    const cluster = pusherConfig.cluster || 'us3';
    const wsUrl = `wss://ws-${cluster}.pusher.com/app/${pusherConfig.appKey}?protocol=7&client=chrome-extension&version=8.4.0`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('[PW-Exec] WebSocket connected to Pusher');
      
      // Subscribe to the channel
      const subscribeMsg = {
        event: 'pusher:subscribe',
        data: {
          channel: pusherConfig.channelName
        }
      };
      
      ws.send(JSON.stringify(subscribeMsg));
      chrome.storage.local.set({ pusherConnected: true });
    };
    
    ws.onmessage = async (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('[PW-Exec] Pusher message received:', message);
        
        // Handle Pusher protocol messages
        if (message.event === 'pusher:connection_established') {
          console.log('[PW-Exec] Pusher connection established');
        }
        else if (message.event === 'pusher:subscription_succeeded') {
          console.log('[PW-Exec] Pusher subscription succeeded');
        }
        // Handle our custom event
        else if (message.event === pusherConfig.eventName) {
          const data = JSON.parse(message.data);
          console.log('[PW-Exec] Received event data:', data);
          
          if (data.action === 'executePlaywrightCode') {
            // Get active tab if tabId not specified
            if (!data.tabId) {
              const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
              if (tabs.length > 0) {
                data.tabId = tabs[0].id;
              } else {
                console.error('[PW-Exec] No active tab found for execution');
                return;
              }
            }
            
            const result = await handleExecutePlaywrightCode(data);
            
            // If a response channel and event are provided, send the result
            if (data.responseChannel && data.responseEvent) {
              // Need to make an API call to trigger an event since we can't publish directly from client
              fetch(data.responseEndpoint || 'https://your-api-endpoint.com/trigger-event', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                  channel: data.responseChannel,
                  event: data.responseEvent,
                  data: {
                    type: 'executionResult',
                    requestId: data.requestId,
                    result
                  }
                })
              }).catch(error => {
                console.error('[PW-Exec] Failed to send result back via API:', error);
              });
            }
          }
        }
      } catch (error) {
        console.error('[PW-Exec] Error processing Pusher message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('[PW-Exec] Pusher WebSocket disconnected');
      chrome.storage.local.set({ pusherConnected: false });
      ws = null;
      
      // Attempt to reconnect after a delay
      if (!wsReconnectTimer && pusherConfig.appKey) {
        wsReconnectTimer = setTimeout(() => {
          connectToPusher(pusherConfig);
        }, 5000); // Reconnect after 5 seconds
      }
    };
    
    ws.onerror = (error) => {
      console.error('[PW-Exec] Pusher WebSocket error:', error);
      chrome.storage.local.set({ pusherConnected: false });
    };
  } catch (error) {
    console.error('[PW-Exec] Failed to connect to Pusher:', error);
    chrome.storage.local.set({ pusherConnected: false });
  }
}

/**
 * Disconnect from Pusher
 */
function disconnectPusher() {
  if (ws) {
    try {
      ws.close();
    } catch (e) {
      console.warn('[PW-Exec] Error disconnecting Pusher WebSocket:', e);
    }
    ws = null;
  }
  
  if (wsReconnectTimer) {
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }
  
  chrome.storage.local.set({ pusherConnected: false });
}

/**
 * Utility to promisify chrome.debugger.sendCommand
 */
function sendDebuggerCommand(tabId, method, params = {}) {
  return new Promise((resolve, reject) => {
    chrome.debugger.sendCommand({ tabId }, method, params, (res) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(res);
      }
    });
  });
}

/**
 * Attach debugger to a tab at protocol version 1.3
 */
function attachDebugger(tabId) {
  return new Promise((resolve, reject) => {
    chrome.debugger.attach({ tabId }, '1.3', () => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve();
      }
    });
  });
}

/**
 * Detach debugger (ignore errors)
 */
function detachDebugger(tabId) {
  return new Promise((resolve) => {
    chrome.debugger.detach({ tabId }, resolve);
  });
}

/**
 * Inject helper (page) API into current execution context
 */
async function injectHelpers(tabId) {
  const helperScript = `
    window.__pw_vars = window.__pw_vars || {};
    window.page = {
      goto: async (url) => {
        location.href = url;
        return new Promise(r => setTimeout(r, 0));
      },
      click: async (sel) => {
        const el = document.querySelector(sel);
        if (!el) throw new Error('Element not found: ' + sel);
        el.click();
      },
      fill: async (sel, val) => {
        const el = document.querySelector(sel);
        if (!el) throw new Error('Element not found: ' + sel);
        el.value = val;
        el.dispatchEvent(new Event('input', {bubbles:true}));
      },
      textContent: async (sel) => {
        const el = document.querySelector(sel);
        if (!el) throw new Error('Element not found: ' + sel);
        return el.textContent;
      },
      innerText: async (sel) => {
        const el = document.querySelector(sel);
        if (!el) throw new Error('Element not found: ' + sel);
        return el.innerText;
      },
      waitForSelector: async (sel, {timeout}= {}) => {
        const ms = timeout || 5000; const start = Date.now();
        return new Promise((resolve, reject)=>{
          const check=()=>{
            const e=document.querySelector(sel);
            if(e) return resolve(e);
            if(Date.now()-start>ms) return reject(new Error('Timeout'));
            setTimeout(check,100);
          };check();
        });
      }
    };
  `;
  await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
    expression: helperScript,
    awaitPromise: false
  });
}

/**
 * Wait for Page.loadEventFired after navigation
 */
function waitForLoadEvent(tabId) {
  return new Promise((resolve) => {
    const listener = (src, method) => {
      if (src.tabId === tabId && method === 'Page.loadEventFired') {
        chrome.debugger.onEvent.removeListener(listener);
        resolve();
      }
    };
    chrome.debugger.onEvent.addListener(listener);
  });
}

/**
 * Execute user code line-by-line handling navigation.
 */
async function runLines(tabId, codeLines) {
  let returnValue;
  for (let i = 0; i < codeLines.length; i++) {
    let rawLine = codeLines[i];
    let line = rawLine.trim();
    if (!line || line.startsWith('//')) continue;

    // Debug: log each line that we attempt to run
    console.log('[PW-Exec] runLines executing:', line);

    // navigation
    const gotoMatch = line.match(/await\s+page\.goto\(['"]([^'"]+)['"]\)/);
    if (gotoMatch) {
      const targetUrl = gotoMatch[1];
      await sendDebuggerCommand(tabId, 'Page.navigate', { url: targetUrl });
      await waitForLoadEvent(tabId);
      // new execution context: re-enable runtime and helpers
      await sendDebuggerCommand(tabId, 'Runtime.enable');
      await injectHelpers(tabId);
      continue;
    }

    // return statement
    if (line.startsWith('return ')) {
      // Handle single-line or multi-line return expressions
      let expr = line.slice(7).trim(); // text after 'return '

      // Remove trailing semicolon (common in single-line returns)
      if (expr.endsWith(';')) {
        expr = expr.slice(0, -1);
      }

      const startsWithBrace = expr.startsWith('{');
      // If the return starts an object literal spanning multiple lines
      if (startsWithBrace && !expr.includes('}')) {
        const collected = [expr];
        let openBraces = 1;
        while (openBraces > 0 && ++i < codeLines.length) {
          const next = codeLines[i];
          const open = (next.match(/\{/g) || []).length;
          const close = (next.match(/\}/g) || []).length;
          openBraces += open - close;
          collected.push(next.trim());
          if (openBraces === 0) break;
        }
        expr = collected.join('\n');
      }
      
      // Instead of trying to rewrite the return expression, use a special function
      // that will evaluate the return expression with access to window.__pw_vars
      const wrappedExpr = `
        (function() {
          // Create local variables from window.__pw_vars for direct access
          const vars = window.__pw_vars || {};
          for (const key in vars) {
            if (Object.prototype.hasOwnProperty.call(vars, key)) {
              eval('var ' + key + ' = vars[key];');
            }
          }
          
          // Now evaluate the return expression with access to those variables
          try {
            return ${expr};
          } catch (error) {
            console.error('Error in return statement:', error.message);
            return { error: error.message };
          }
        })()
      `;
      
      console.log('[PW-Exec] Wrapped return expression:', wrappedExpr);

      try {
        const res = await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
          expression: wrappedExpr,
          returnByValue: true
        });
        if (res && res.result) {
          returnValue = res.result.value;
        } else if (res && res.exceptionDetails) {
          console.warn('[PW-Exec] Evaluation exception in return statement:', res.exceptionDetails.text || res.exceptionDetails);
          await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
            expression: `console.error('[Line ${i+1}] Return statement error: ${res.exceptionDetails.text || "Unknown error"}')`,
            awaitPromise: false
          });
        }
        console.log('[PW-Exec] return expression evaluated to:', returnValue);
      } catch (error) {
        console.error('[PW-Exec] Error evaluating return statement:', error);
        await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
          expression: `console.error('[Line ${i+1}] Return statement error: ${error.message}')`,
          awaitPromise: false
        });
      }
      break;
    }

    // Handle variable declarations
    let originalLine = line;
    let processedLine = line;
    
    try {
      if (/^\s*(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*(?:=.*|;)?$/.test(line)) {
        const declarationMatch = line.match(/^\s*(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*(?:=(.*))?;?$/);
        if (declarationMatch) {
          const varName = declarationMatch[1];
          const valueExpr = declarationMatch[2] ? declarationMatch[2].trim() : 'undefined';
          processedLine = `window.__pw_vars.${varName} = ${valueExpr};`;
          console.log('[PW-Exec] Variable declaration processed:', { original: originalLine, processed: processedLine });
        }
      } else {
        // Only rewrite variable references in non-declaration lines
        processedLine = line.replace(/\b([a-zA-Z_$][\w$]*)\b/g, (match, varName) => {
          // Skip keywords and global objects we obviously don't want to rewrite
          const blacklist = ['await', 'page', 'document', 'window', 'location', 'console', 'return', 'if', 'else', 'for', 'while', 'switch', 'break', 'continue', 'function', 'async', 'let', 'const', 'var', 'true', 'false', 'null', 'undefined'];
          if (blacklist.includes(match)) return match;
          return `(window.__pw_vars.${varName} !== undefined ? window.__pw_vars.${varName} : ${varName})`;
        });
      }

      // regular line, evaluate with error handling
      const clickNav = /await\s+page\.click\(/.test(processedLine);
      
      console.log('[PW-Exec] Executing processed line:', processedLine);
      
      // Execute the line with detailed error handling
      try {
        const result = await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
          expression: `(async ()=>{ 
            try { 
              ${processedLine} 
              return { success: true };
            } catch (error) { 
              console.error('[Line ${i+1}] Error executing: ${originalLine.replace(/'/g, "\\'")}\\nError: ' + error.message);
              return { success: false, error: error.message };
            }
          })()`,
          awaitPromise: true,
          returnByValue: true
        });
        
        if (result && result.result && result.result.value) {
          const execResult = result.result.value;
          if (!execResult.success) {
            console.error(`[PW-Exec] Line ${i+1} execution failed:`, execResult.error);
          }
        } else if (result && result.exceptionDetails) {
          console.error(`[PW-Exec] Line ${i+1} evaluation exception:`, result.exceptionDetails.text || result.exceptionDetails);
          await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
            expression: `console.error('[Line ${i+1}] Evaluation error: ${result.exceptionDetails.text || "Unknown error"}')`,
            awaitPromise: false
          });
        }
      } catch (error) {
        console.error(`[PW-Exec] Line ${i+1} execution error:`, error);
        await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
          expression: `console.error('[Line ${i+1}] Execution error: ${error.message}')`,
          awaitPromise: false
        });
      }

      // If the line was a click that might trigger navigation, wait briefly
      if (clickNav) {
        const navHappened = await Promise.race([
          waitForLoadEvent(tabId).then(()=>true),
          new Promise(r=>setTimeout(()=>r(false), 10000)) // 10-second fallback
        ]);
        if (navHappened) {
          await sendDebuggerCommand(tabId, 'Runtime.enable');
          await injectHelpers(tabId);
        }
      }
    } catch (lineProcessingError) {
      console.error(`[PW-Exec] Line ${i+1} processing error:`, lineProcessingError);
      await sendDebuggerCommand(tabId, 'Runtime.evaluate', {
        expression: `console.error('[Line ${i+1}] Processing error: ${lineProcessingError.message}')`,
        awaitPromise: false
      });
    }
  }
  return returnValue;
}

/**
 * Handle the executePlaywrightCode action
 */
async function handleExecutePlaywrightCode(req) {
  const tabId = req.tabId;
  const code = req.code || '';
  const logs = [];

  try {
    await attachDebugger(tabId);
    await sendDebuggerCommand(tabId, 'Runtime.enable');
    await sendDebuggerCommand(tabId, 'Page.enable');

    // Capture console messages
    const consoleListener = (src, method, params) => {
      if (src.tabId === tabId && method === 'Runtime.consoleAPICalled') {
        const txt = params.args.map(a => a.value ?? a.description).join(' ');
        logs.push([params.type, txt]);
      }
    };
    chrome.debugger.onEvent.addListener(consoleListener);

    await injectHelpers(tabId);

    const result = await runLines(tabId, code.split('\n'));
    console.log('[PW-Exec] Final result from runLines:', result);

    chrome.debugger.onEvent.removeListener(consoleListener);
    await detachDebugger(tabId);

    // Persist the outcome for potential retrieval by the popup in case the
    // message channel is lost (e.g., due to long navigation).
    const record = { result, logs, ts: Date.now() };
    chrome.storage.local.set({ lastExecution: record });

    // Maintain an execution history list (max 20)
    chrome.storage.local.get(['history'], (data) => {
      const history = Array.isArray(data.history) ? data.history : [];
      history.push(record);
      // Keep newest last, trim size
      const trimmed = history.slice(-20);
      chrome.storage.local.set({ history: trimmed });
    });

    // If response channel and event are provided, send the result back via Pusher
    if (req.responseChannel && req.responseEvent) {
      console.log('[PW-Exec] Response channel and event detected:', { 
        channel: req.responseChannel, 
        event: req.responseEvent,
        requestId: req.requestId
      });
      
      // Send the result back to Pusher
      await sendResultToPusher(
        req.responseChannel, 
        req.responseEvent, 
        {
          requestId: req.requestId,
          result: result,
          logs: logs
        },
        pusherConfig.appKey || '271b88729ca02b9f059d',  // Default to hardcoded key if not set
        pusherConfig.cluster || 'us3'                   // Default to us3 cluster if not set
      );
    } else {
      console.log('[PW-Exec] No response channel/event provided in request');
    }

    return { success: true, result, logs };
  } catch (e) {
    console.error(e);
    try { await detachDebugger(tabId); } catch (_) {/* ignore */}
    return { success: false, error: e.message, logs };
  }
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((req, _sender, sendResponse) => {
  console.log('[PW-Exec] onMessage:', req);
  
  if (req.action === 'executePlaywrightCode') {
    // Run the async workflow in a detached promise chain so we can return synchronously
    (async () => {
      const result = await handleExecutePlaywrightCode(req);
      sendResponse(result);
    })();
    
    // Return true synchronously to keep the message channel open
    return true;
  }
  else if (req.action === 'connectPusher') {
    connectToPusher(req.config);
    sendResponse({ success: true });
    return false;
  }
  else if (req.action === 'disconnectPusher') {
    disconnectPusher();
    sendResponse({ success: true });
    return false;
  }
  else if (req.action === 'checkPusherStatus') {
    const connected = ws !== null && ws.readyState === WebSocket.OPEN;
    sendResponse({ 
      connected, 
      config: pusherConfig 
    });
    return false;
  }
});

// Check if we have saved Pusher config to connect on startup
chrome.storage.local.get(['pusherConfig'], (data) => {
  if (data.pusherConfig) {
    connectToPusher(data.pusherConfig);
  }
}); 