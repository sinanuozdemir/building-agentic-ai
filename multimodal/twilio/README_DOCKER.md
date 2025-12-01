# Running Twilio Voice Assistant with Docker

## Quick Start

1. **Create a `.env` file** from the example:
   ```bash
   cp env.example .env
   ```

2. **Edit `.env`** and add your API keys:
   ```bash
   GROQ_API_KEY=your_actual_groq_api_key_here
   PORT=5015
   NGROK_AUTHTOKEN=your_ngrok_authtoken_here
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - **Local**: http://localhost:5015
   - **Public (via ngrok)**: https://twilio-applied-ai.ngrok.io
   - **API Docs**: http://localhost:5015/docs
   - **Recordings**: http://localhost:5015/recordings
   - **Webhook URL for Twilio**: https://twilio-applied-ai.ngrok.io/incoming-call
   - **WebSocket endpoint**: wss://twilio-applied-ai.ngrok.io/media-stream

## Running Without Docker

If you prefer to run locally:

1. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file** with your API keys:
   ```bash
   cp env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

4. **Run the application**:
   ```bash
   python3 twilio_app.py
   ```
   Or with uvicorn directly:
   ```bash
   uvicorn twilio_app:app --host 0.0.0.0 --port 5015 --reload
   ```

## Environment Variables

- `GROQ_API_KEY` (required): Your Groq API key from https://console.groq.com/
- `PORT` (optional): Server port, defaults to 5015
- `NGROK_AUTHTOKEN` (required for ngrok): Your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken

## Ngrok Configuration

The Docker Compose setup includes an ngrok service that automatically creates a public tunnel to your local application:

- **Custom Domain**: `twilio-applied-ai.ngrok.io` (requires paid ngrok account)
- **Region**: EU
- **Tunnels to**: `twilio-app:5015` (internal Docker network)

**Note**: To use a custom domain like `twilio-applied-ai.ngrok.io`, you need:
1. A paid ngrok account
2. The domain configured in your ngrok dashboard
3. Your `NGROK_AUTHTOKEN` set in the `.env` file

If you don't have a paid account, you can modify the ngrok service in `docker-compose.yml` to remove the `--url` flag and use a free random domain.

## Docker Commands

- **Start**: `docker-compose up`
- **Start in background**: `docker-compose up -d`
- **Stop**: `docker-compose down`
- **View logs**: `docker-compose logs -f`
- **View ngrok logs**: `docker-compose logs -f ngrok`
- **View app logs**: `docker-compose logs -f twilio-app`
- **Rebuild**: `docker-compose up --build`

## Notes

- Recorded calls are saved in the `recorded_calls/` directory
- The `.env` file is gitignored, so your API keys won't be committed
- Make sure your `.env` file is in the same directory as `docker-compose.yml`

