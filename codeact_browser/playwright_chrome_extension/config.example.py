# Configuration file for Pusher secrets (EXAMPLE)
# Copy this file to config.py and fill in your actual secrets

# Pusher credentials
PUSHER_CONFIG = {
    "APP_ID": "YOUR_PUSHER_APP_ID",
    "APP_KEY": "YOUR_PUSHER_APP_KEY",
    "APP_SECRET": "YOUR_PUSHER_APP_SECRET",
    "CLUSTER": "YOUR_PUSHER_CLUSTER"  # e.g., "us3", "eu", "ap1"
}

# Event details
CHANNEL = "my-channel"
EVENT = "my-event"
RESPONSE_CHANNEL = "response-channel"
RESPONSE_EVENT = "execution-result" 