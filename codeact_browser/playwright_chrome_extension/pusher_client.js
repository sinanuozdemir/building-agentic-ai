// A simple standalone example to send a message to Pusher with proper authentication
// This demonstrates how to properly authenticate including body_md5

async function sendPusherMessage() {
  // Pusher credentials (same as those used by the extension)
  const APP_ID = "1987489";
  const APP_KEY = "271b88729ca02b9f059d";
  const APP_SECRET = "dd35f6207be6a4587204";  // Replace with your actual secret
  const CLUSTER = "us3";
  
  // Message details
  const CHANNEL = "response-channel";
  const EVENT = "execution-result";
  
  // Create the data payload
  const data = {
    requestId: "test-" + Date.now(),
    result: {
      finalURL: "https://example.com",
      finalTitle: "Example Domain"
    },
    logs: [
      ["log", "This is a test log message"]
    ]
  };
  
  // Format the data for Pusher
  const requestBody = {
    name: EVENT,
    channel: CHANNEL,
    data: JSON.stringify(data)  // Pusher requires data to be a string
  };
  
  // Generate timestamp for auth
  const timestamp = Math.floor(Date.now() / 1000);
  
  // Calculate MD5 hash of the request body (required by Pusher)
  const bodyString = JSON.stringify(requestBody);
  const bodyMD5 = await calculateMD5(bodyString);
  
  // Authentication parameters
  const authParams = {
    auth_key: APP_KEY,
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
  const stringToSign = `POST\n/apps/${APP_ID}/events\n${queryString}`;
  
  // Generate the signature
  const signature = await calculateHMACSHA256(stringToSign, APP_SECRET);
  
  // Full URL with authentication parameters
  const url = `https://api-${CLUSTER}.pusher.com/apps/${APP_ID}/events?${queryString}&auth_signature=${signature}`;
  
  console.log('Sending message to Pusher...');
  console.log('URL:', url);
  console.log('Body:', bodyString);
  
  // Make the request
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: bodyString
    });
    
    if (response.ok) {
      console.log('✅ Message sent successfully to Pusher!');
      const responseText = await response.text();
      console.log('Response:', responseText);
    } else {
      console.error('❌ Failed to send message to Pusher:', response.status);
      const errorText = await response.text();
      console.error('Error details:', errorText);
    }
  } catch (error) {
    console.error('❌ Error sending message to Pusher:', error);
  }
}

// Helper function to calculate MD5 hash
async function calculateMD5(message) {
  // Using SubtleCrypto API (available in modern browsers and service workers)
  const msgUint8 = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('MD5', msgUint8);
  
  // Convert to hex string
  return Array.from(new Uint8Array(hashBuffer))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
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

// Execute the function
sendPusherMessage(); 