// Configuration file for secrets and other settings (EXAMPLE)
// Copy this file to config.js and fill in your actual secrets

const PUSHER_CONFIG = {
  APP_ID: "YOUR_PUSHER_APP_ID",
  APP_KEY: "YOUR_PUSHER_APP_KEY",
  APP_SECRET: "YOUR_PUSHER_APP_SECRET",
  CLUSTER: "YOUR_PUSHER_CLUSTER" // e.g., "us3", "eu", "ap1"
};

// Export the configuration
self.PUSHER_CONFIG = PUSHER_CONFIG; 