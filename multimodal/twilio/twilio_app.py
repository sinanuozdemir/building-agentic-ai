import os
import json
import base64
import asyncio
import wave
import struct
import tempfile
import audioop
import math
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
import logging
from itertools import product
import numpy as np
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress noisy debug logs from external libraries
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("groq._response").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("pydub").setLevel(logging.WARNING)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)

if not SCIPY_AVAILABLE:
    logger.warning("scipy not available, using audioop for resampling (lower quality)")

if not PYDUB_AVAILABLE:
    logger.warning("pydub not available, MP3 saving will be disabled")

# Configuration
PORT = int(os.getenv('PORT', 5015))

# Initialize Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file or environment.")
client = Groq(api_key=groq_api_key)

# Create FastAPI app instance
app = FastAPI(
    title="Twilio Voice Assistant with Audio Recording",
    description="A FastAPI application for handling Twilio voice calls with WebSocket media streaming and audio recording capabilities.",
    version="1.0.0"
)

# Audio storage configuration
AUDIO_STORAGE_DIR = "recorded_calls"
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)

# Global storage for call audio data
call_audio_buffers = {}

# Global storage for transcription buffers (separate from recording buffers)
transcription_buffers = {}
last_audio_time = {}

# Global storage for decibel level history
decibel_history = {}

# Track last transcription time to prevent duplicate transcriptions
last_transcription_time = {}

def mulaw_to_linear(mulaw_data):
    """Convert μ-law encoded audio to linear PCM"""
    linear_data = []
    for byte in mulaw_data:
        # μ-law to linear conversion
        byte = ~byte
        sign = (byte & 0x80) >> 7
        exponent = (byte & 0x70) >> 4
        mantissa = byte & 0x0F
        
        if exponent == 0:
            sample = mantissa << 4
        else:
            sample = ((mantissa | 0x10) << (exponent + 3))
        
        if sign:
            sample = -sample
        
        linear_data.append(sample)
    
    return linear_data

def save_call_audio(call_sid, audio_buffer):
    """Save accumulated audio buffer as WAV file"""
    if not audio_buffer:
        logger.warning("No audio data to save for call %s", call_sid)
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"call_{call_sid}_{timestamp}.wav"
    filepath = os.path.join(AUDIO_STORAGE_DIR, filename)
    
    try:
        # Convert μ-law to linear PCM
        linear_audio = mulaw_to_linear(audio_buffer)
        
        # Create WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(8000)  # 8kHz sample rate
            
            # Convert to 16-bit signed integers and write
            audio_data = struct.pack('<' + 'h' * len(linear_audio), *linear_audio)
            wav_file.writeframes(audio_data)
        
        logger.info("Saved audio file: %s (%d bytes of audio)", filepath, len(audio_buffer))
        
    except Exception as e:
        logger.error("Failed to save audio file %s: %s", filepath, e)

def save_raw_mulaw_audio(call_sid, audio_buffer):
    """Save raw μ-law audio data"""
    if not audio_buffer:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"call_{call_sid}_{timestamp}.ulaw"
    filepath = os.path.join(AUDIO_STORAGE_DIR, filename)
    
    try:
        with open(filepath, 'wb') as f:
            f.write(bytes(audio_buffer))
        logger.info("Saved raw μ-law file: %s (%d bytes)", filepath, len(audio_buffer))
    except Exception as e:
        logger.error("Failed to save raw audio file %s: %s", filepath, e)

def save_audio_as_mp3(call_sid, audio_buffer, suffix=""):
    """Save audio buffer as MP3 file"""
    if not audio_buffer:
        return
    
    if not PYDUB_AVAILABLE:
        return
    
    # Check if ffmpeg is available (required by pydub for MP3)
    import shutil
    if not shutil.which('ffmpeg'):
        logger.error("🔊 [MP3] ffmpeg not found! Install with: apt-get install ffmpeg")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"call_{call_sid}_{timestamp}{suffix_str}.mp3"
    filepath = os.path.join(AUDIO_STORAGE_DIR, filename)
    
    try:
        # Convert μ-law to linear PCM
        linear_audio = mulaw_to_linear(audio_buffer)
        
        # Create temporary WAV file
        tmp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
                
                with wave.open(tmp_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(8000)  # 8kHz sample rate
                    
                    # Convert to 16-bit signed integers and write
                    audio_data = struct.pack('<' + 'h' * len(linear_audio), *linear_audio)
                    wav_file.writeframes(audio_data)
                
                # Convert WAV to MP3 using pydub
                audio_segment = AudioSegment.from_wav(tmp_wav_path)
                audio_segment.export(filepath, format="mp3", bitrate="128k")
                
                mp3_size = os.path.getsize(filepath)
                logger.info("🔊 [MP3] Saved: %s (%.2f MB)", filename, mp3_size / (1024*1024))
                
        finally:
            # Clean up temp WAV file
            if tmp_wav_path and os.path.exists(tmp_wav_path):
                try:
                    os.unlink(tmp_wav_path)
                except Exception:
                    pass
        
    except Exception as e:
        logger.error("🔊 [MP3] Failed to save: %s", e)

async def transcribe_audio(call_sid, audio_buffer, model="whisper-large-v3-turbo", sample_rate=8000):
    """Transcribe audio buffer to text using Groq with configurable sample rate"""
    if not audio_buffer or len(audio_buffer) < 1600:  # At least 100ms of audio
        return None
    
    try:
        logger.info("Transcribing audio for call %s (%d bytes) at %d Hz", call_sid, len(audio_buffer), sample_rate)
        
        # Convert μ-law to linear PCM
        linear_audio = mulaw_to_linear(audio_buffer)
        
        # Create temporary WAV file with specified sample rate
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(8000)  # Original μ-law is 8kHz
                
                # Convert to 16-bit signed integers
                audio_data = struct.pack('<' + 'h' * len(linear_audio), *linear_audio)
                wav_file.writeframes(audio_data)
            
            # If we want a different sample rate, resample the audio
            if sample_rate != 8000:
                # Read the 8kHz audio and resample
                logger.info("Resampling audio to %d Hz", sample_rate)
                with wave.open(tmp_file.name, 'rb') as original_wav:
                    original_frames = original_wav.readframes(original_wav.getnframes())
                    
                # Resample using audioop
                import audioop
                resampled_frames = audioop.ratecv(original_frames, 2, 1, 8000, sample_rate, None)[0]
                
                # Write resampled audio to new temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as resampled_file:
                    with wave.open(resampled_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(resampled_frames)
                    
                    # Clean up original temp file
                    os.unlink(tmp_file.name)
                    transcription_file = resampled_file.name
            else:
                transcription_file = tmp_file.name
            
            # Transcribe with Groq using distil-whisper-large-v3-en
            with open(transcription_file, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(transcription_file, audio_file.read()),
                    model=model,
                    response_format="text"
                )
            
            # Clean up temp file
            os.unlink(transcription_file)
            
            if transcription and transcription.strip():
                logger.info("Transcription for call %s: %s", call_sid, transcription.strip())
                return transcription.strip()
            else:
                logger.debug("Empty transcription for call %s", call_sid)
                return None
                
    except Exception as e:
        logger.error("Transcription failed for call %s: %s", call_sid, e)
        return None

async def process_transcription_and_respond(websocket, stream_sid, call_sid, audio_buffer, sample_rate=8000):
    """Process transcription and send audio response back to caller"""
    try:
        sample_rates_to_try = [sample_rate]
        models_to_try = ["whisper-large-v3-turbo"]
        best_transcription = None
        
        for rate, model in product(sample_rates_to_try, models_to_try):
            try:
                transcription = await transcribe_audio(call_sid, audio_buffer, model=model, sample_rate=rate)
                if transcription and len(transcription.strip()) > 0:
                    # Filter out common silence/noise artifacts from Whisper
                    transcription_clean = transcription.strip()
                    
                    # Ignore transcriptions that are just punctuation or very short noise
                    ignore_patterns = [
                        ".", "..", "...","you", "you.", 
                        "I heard you say", "you said", "got it", "repeating back"
                    ]
                    
                    # Check if transcription is just noise/punctuation
                    if transcription_clean in ignore_patterns:
                        logger.info("🎤 [PROCESS] Ignoring noise/punctuation: '%s'", transcription_clean)
                        continue
                    
                    # Ignore very short transcriptions (likely noise)
                    if len(transcription_clean) < 3:
                        logger.info("🎤 [PROCESS] Ignoring too short transcription: '%s'", transcription_clean)
                        continue
                    
                    logger.info("🎤 [PROCESS] Caller said: %s", transcription_clean)
                    best_transcription = transcription_clean
                    break
            except Exception as rate_error:
                logger.warning("🎯 [PROCESS] Transcription failed: %s", rate_error)
                continue
        
        # Save audio as MP3
        save_audio_as_mp3(call_sid, audio_buffer, suffix="transcription")
        
        # If we got a transcription, parrot it back to the user
        if best_transcription:
            response_text = f"I heard you say: {best_transcription}"
            await send_audio_response(websocket, stream_sid, response_text)
        
    except Exception as e:
        logger.error("🎯 [PROCESS] Error: %s", e)

def convert_wav_to_mulaw(wav_data):
    """Convert WAV audio data to μ-law format for Twilio"""
    try:
        # Read WAV data
        with wave.open(wav_data, 'rb') as wav_file:
            # Get audio parameters
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()
            duration_seconds = num_frames / sample_rate if sample_rate > 0 else 0
            
            # Read all frames
            frames = wav_file.readframes(num_frames)
            
            original_sample_rate = sample_rate
            original_sample_width = sample_width
            
            # Convert to mono if stereo
            if num_channels == 2:
                frames = audioop.tomono(frames, sample_width, 1, 1)
            
            # Ensure we have 16-bit samples before resampling
            if sample_width != 2:
                if sample_width == 1:
                    frames = audioop.lin2lin(frames, 1, 2)
                elif sample_width == 3:
                    frames = audioop.lin2lin(frames, 3, 2)
                elif sample_width == 4:
                    frames = audioop.lin2lin(frames, 4, 2)
                sample_width = 2
            
            # Resample to 8kHz if needed (Twilio requires 8kHz)
            if sample_rate != 8000:
                resample_ratio = 8000 / sample_rate
                input_samples = len(frames) // sample_width
                target_samples = int(input_samples * resample_ratio)
                
                logger.warning("🔄 [WAV→μLAW] Resampling %d Hz → 8000 Hz (%.1fx downsampling)", 
                              sample_rate, 1/resample_ratio)
                
                if SCIPY_AVAILABLE:
                    # Use scipy for high-quality resampling with anti-aliasing
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    
                    # For downsampling, use decimate or resample_poly with anti-aliasing
                    # decimate applies an anti-aliasing filter automatically
                    if sample_rate % 8000 == 0:
                        # If sample_rate is a multiple of 8000, use decimate (best quality)
                        decimation_factor = sample_rate // 8000
                        resampled_array = signal.decimate(audio_array, decimation_factor, ftype='iir', zero_phase=True)
                    else:
                        # Otherwise use resample_poly which handles anti-aliasing better than resample
                        # resample_poly uses polyphase filtering which is better for downsampling
                        up = 1
                        down = int(sample_rate / 8000)
                        # Simplify the ratio
                        gcd_val = np.gcd(up, down)
                        up //= gcd_val
                        down //= gcd_val
                        resampled_array = signal.resample_poly(audio_array, up, down, padtype='line')
                    
                    # Clamp values to int16 range and convert
                    resampled_array = np.clip(resampled_array, -32768, 32767)
                    resampled_array = resampled_array.astype(np.int16)
                    frames = resampled_array.tobytes()
                else:
                    # Fallback to audioop (lower quality)
                    logger.warning("🔄 [WAV→μLAW] scipy not available, using lower quality resampling")
                    resampled_result = audioop.ratecv(frames, sample_width, 1, sample_rate, 8000, None)
                    frames = resampled_result[0]
            
            # Convert to μ-law (requires 16-bit input)
            mulaw_data = audioop.lin2ulaw(frames, sample_width)
            
            logger.debug("🔄 [WAV→μLAW] Converted: %d Hz/%d-bit/%dch → 8000 Hz/16-bit/mono → %d bytes μ-law", 
                        original_sample_rate, original_sample_width * 8, num_channels, len(mulaw_data))
            
            return mulaw_data
    except Exception as e:
        logger.error("🔄 [WAV→μLAW] Conversion failed: %s", e)
        return None

async def send_audio_chunks(websocket: WebSocket, stream_sid: str, audio_data: bytes):
    """Send audio data in chunks to Twilio"""
    try:
        # Convert audio to μ-law format
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(audio_data)
                tmp_file.flush()
            
            mulaw_data = convert_wav_to_mulaw(tmp_file_path)
            
            if not mulaw_data:
                logger.error("📤 [SEND AUDIO] Failed to convert audio to μ-law")
                await send_silence(websocket, stream_sid)
                return
            
            # Send audio in 160-byte chunks (20ms at 8kHz)
            # Send chunks faster than real-time to build a buffer and prevent stuttering
            chunk_size = 160
            chunks_sent = 0
            
            for i in range(0, len(mulaw_data), chunk_size):
                chunk = mulaw_data[i:i + chunk_size]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk += bytes([0xFF] * (chunk_size - len(chunk)))
                
                # Encode as base64
                chunk_payload = base64.b64encode(chunk).decode('utf-8')
                
                audio_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": chunk_payload
                    }
                }
                
                try:
                    if websocket.client_state.name != "CONNECTED":
                        break
                        
                    await websocket.send_json(audio_message)
                    chunks_sent += 1
                    
                    # Send chunks much faster than real-time (1ms delay = 20x faster) to build large buffer
                    # This prevents stuttering by ensuring there's always plenty of audio buffered
                    await asyncio.sleep(0.001)  # 1ms delay - send as fast as possible
                except Exception as e:
                    logger.error("📤 [SEND AUDIO] Failed to send chunk: %s", e)
                    break
            
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
            
    except Exception as e:
        logger.error("📤 [SEND AUDIO] Failed: %s", e)
        await send_silence(websocket, stream_sid)

async def send_silence(websocket: WebSocket, stream_sid: str, duration_ms: int = 200):
    """Send silence as fallback or keep-alive
    
    Args:
        websocket: WebSocket connection
        stream_sid: Stream SID
        duration_ms: Duration of silence in milliseconds (default 200ms)
    """
    logger.info("Sending silence (duration: %d ms)", duration_ms)
    
    # Generate silence (μ-law encoded zeros)
    # μ-law silence is 0xFF (255 in decimal)
    silence_chunk = bytes([0xFF] * 160)  # 160 bytes of silence (20ms at 8kHz)
    silence_payload = base64.b64encode(silence_chunk).decode('utf-8')
    
    # Calculate number of chunks needed for the duration
    # Each chunk is 20ms, so for duration_ms we need duration_ms / 20 chunks
    num_chunks = max(1, duration_ms // 20)
    
    # Send multiple chunks of silence
    for i in range(num_chunks):
        audio_message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": silence_payload
            }
        }
        
        try:
            # Check if WebSocket is still connected
            if websocket.client_state.name != "CONNECTED":
                logger.warning("WebSocket not connected, stopping silence send")
                break
                
            await websocket.send_json(audio_message)
            await asyncio.sleep(0.02)  # 20ms delay between chunks
        except Exception as e:
            logger.error("Failed to send silence chunk: %s", e)
            break

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {
        "message": "Twilio Media Stream Server with Audio Recording",
        "status": "running",
        "features": {
            "incoming_calls": "/incoming-call",
            "media_stream": "/media-stream", 
            "recordings_list": "/recordings",
            "recording_download": "/recordings/<filename>",
            "test_endpoint": "/test"
        }
    }
@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    logger.info("Incoming call received")
    
    response = VoiceResponse()
    
    # Get the host from the request to build the WebSocket URL
    # Check X-Forwarded-Proto header for protocol when behind proxy (ngrok)
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    if forwarded_proto == "https" or request.url.scheme == "https":
        protocol = "wss"
    else:
        protocol = "ws"
    
    host = request.headers.get("host", request.url.hostname)
    
    # Add a brief pause to ensure call stays alive while WebSocket connects
    # This gives time for the WebSocket connection to establish
    response.pause(length=1)
    
    # Connect to our media stream WebSocket
    connect = Connect()
    connect.stream(url=f'{protocol}://{host}/media-stream')
    response.append(connect)
    
    logger.info("Returning TwiML response with WebSocket URL: %s://%s/media-stream", protocol, host)
    logger.info("Request headers: %s", dict(request.headers))
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections for Twilio Media Streams."""
    logger.info("Media stream WebSocket connection initiated")
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted successfully")
    except Exception as e:
        logger.error("Failed to accept WebSocket connection: %s", e, exc_info=True)
        return
    
    stream_sid = None
    call_sid = None
    has_seen_media = False
    message_count = 0

    try:
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                event_type = data.get('event')
                
                if event_type != "media":
                    logger.info("Received event: %s", event_type)
                
                if event_type == "connected":
                    logger.info("Media stream connected")
                    
                elif event_type == "start":
                    stream_sid = data['start']['streamSid']
                    call_sid = data['start']['callSid']
                    logger.info("Stream started - SID: %s, Call: %s", stream_sid, call_sid)
                    
                    # Initialize audio buffers for this call
                    call_audio_buffers[call_sid] = []  # For full recording
                    transcription_buffers[call_sid] = []  # For real-time transcription
                    last_audio_time[call_sid] = time.time()
                    decibel_history[call_sid] = []
                    
                    # Immediately send silence to keep the connection alive
                    # This prevents Twilio from hanging up while we prepare the greeting
                    # Send 2 seconds of silence to give TTS time to generate
                    logger.info("Sending initial silence to keep connection alive")
                    try:
                        asyncio.create_task(send_silence(websocket, stream_sid, duration_ms=2000))
                    except Exception as e:
                        logger.error("Failed to send initial silence: %s", e)
                    
                    # Send a simple greeting response to acknowledge the call
                    # Use create_task to avoid blocking the WebSocket message loop
                    try:
                        logger.info("Starting greeting TTS task")
                        asyncio.create_task(send_audio_response(websocket, stream_sid, "Hello! How can I help you today?"))
                    except Exception as e:
                        logger.error("Failed to start greeting task: %s", e, exc_info=True)
                        # Send more silence as fallback to keep connection alive
                        try:
                            asyncio.create_task(send_silence(websocket, stream_sid))
                        except Exception as silence_error:
                            logger.error("Failed to send silence fallback: %s", silence_error)
                
                elif event_type == "media":
                    # Extract and save audio data
                    payload = data.get('media', {}).get('payload', '')
                    if payload and call_sid:
                        try:
                            # Decode base64 audio data
                            audio_chunk = base64.b64decode(payload)
                            
                            # Add to both recording and transcription buffers
                            call_audio_buffers[call_sid].extend(audio_chunk)
                            transcription_buffers[call_sid].extend(audio_chunk)
                            
                            # Update last audio time
                            current_time = time.time()
                            previous_audio_time = last_audio_time.get(call_sid, current_time)
                            last_audio_time[call_sid] = current_time
                            
                            if not has_seen_media:
                                logger.info("🎤 [AUDIO] Started recording audio for call %s", call_sid)
                                logger.info("🎤 [AUDIO] First audio chunk: %d bytes", len(audio_chunk))
                                has_seen_media = True
                            
                            # Detect silence and transcribe when appropriate
                            # We need at least 1 second of audio (8000 samples = 8000 bytes at 8kHz) before transcribing
                            transcription_buffer_size = len(transcription_buffers[call_sid])
                            recording_buffer_size = len(call_audio_buffers[call_sid])
                            
                            logger.debug("🎤 [AUDIO] Buffer sizes - transcription: %d bytes, recording: %d bytes", 
                                       transcription_buffer_size, recording_buffer_size)
                            
                            # --- Calculate decibel level for this chunk ---
                            # μ-law to linear PCM conversion for dB calculation
                            # Each byte in audio_chunk is a μ-law sample
                            import audioop
                            try:
                                # Convert μ-law to linear PCM (16-bit signed)
                                linear_pcm = audioop.ulaw2lin(audio_chunk, 2)
                                # Calculate RMS (root mean square)
                                rms = audioop.rms(linear_pcm, 2)
                                # Avoid log(0) by ensuring rms > 0
                                silence = False
                                if rms > 0:
                                    decibels = 20 * math.log10(rms / 32768)
                                    # Silence threshold: only very quiet background noise (< -80 dB)
                                    # Normal phone speech is around -68 to -74 dB, so this won't classify speech as silence
                                    silence = decibels < -80  # Much lower threshold - only true silence
                                else:
                                    # decibels = -float('inf')
                                    silence = True
                                    decibels = -200  # TODO this is fake
                                # Only log dBFS occasionally to reduce noise
                                if transcription_buffer_size % 8000 == 0:  # Log every ~1 second
                                    logger.debug("🎤 [AUDIO] Audio chunk dBFS: %.2f dB (silence=%s)", decibels, silence)
                            except Exception as db_e:
                                logger.warning("🎤 [AUDIO] Could not calculate dBFS: %s", db_e)
                                decibels = None
                            # --- End decibel calculation ---

                            # --- Keep track of last second's worth of decibel readings ---
                            # We'll keep a list of (timestamp, decibels) for each call_sid
                            now = time.time()
                            if call_sid not in decibel_history:
                                decibel_history[call_sid] = []
                            if decibels is not None:
                                decibel_history[call_sid].append((now, decibels))
                            # Remove entries older than 1 second
                            one_sec_ago = now - 1.0
                            decibel_history[call_sid] = [
                                (ts, db) for (ts, db) in decibel_history[call_sid] if ts >= one_sec_ago
                            ]
                            # Optionally, you can compute the average dBFS over the last second:
                            avg_dbfs = None
                            if decibel_history[call_sid]:
                                avg_dbfs = sum(db for (_, db) in decibel_history[call_sid]) / len(decibel_history[call_sid])
                                # Only log occasionally to reduce noise
                                if transcription_buffer_size % 8000 == 0:  # Log every ~1 second
                                    logger.debug("🎤 [AUDIO] Average dBFS over last second: %.2f dB (from %d samples)", 
                                               avg_dbfs, len(decibel_history[call_sid]))
                            
                            # Minimum buffer size for transcription: ~1 second of audio (8000 bytes at 8kHz)
                            MIN_TRANSCRIPTION_BUFFER_SIZE = 8000
                            
                            # Only check for transcription if we have enough audio accumulated
                            if transcription_buffer_size >= MIN_TRANSCRIPTION_BUFFER_SIZE:
                                # Check if we've had sustained silence (average dBFS below threshold for last second)
                                # AND we haven't transcribed recently (to avoid duplicate transcriptions)
                                # Silence threshold: -80 dB (only very quiet background noise)
                                # Normal phone speech is around -68 to -74 dB, so this correctly identifies silence
                                SILENCE_THRESHOLD = -80
                                if avg_dbfs is not None and avg_dbfs < SILENCE_THRESHOLD:
                                    # Check if we've had silence for at least 0.5 seconds
                                    # Count how many recent chunks were silent
                                    recent_silent_chunks = sum(1 for (_, db) in decibel_history[call_sid] if db < SILENCE_THRESHOLD)
                                    total_recent_chunks = len(decibel_history[call_sid])
                                    
                                    # Check if there was ANY speech before the silence
                                    # Speech threshold: -70 dB (normal phone speech is -68 to -74 dB)
                                    speech_threshold = -70  # Matches actual phone speech levels
                                    had_speech = False
                                    if len(decibel_history[call_sid]) > 10:  # Need enough history
                                        # Check if any chunk in the last 2 seconds had speech-level audio
                                        for ts, db in decibel_history[call_sid]:
                                            if db > speech_threshold:
                                                had_speech = True
                                                break
                                    
                                    # Only transcribe if:
                                    # 1. Most of the last second was silence (>80%)
                                    # 2. There was actual speech detected before the silence (not just silence the whole time)
                                    if total_recent_chunks > 0 and (recent_silent_chunks / total_recent_chunks) > 0.8:
                                        if had_speech:
                                            # Debounce: Don't transcribe if we transcribed less than 2 seconds ago
                                            last_trans_time = last_transcription_time.get(call_sid, 0)
                                            time_since_last_transcription = current_time - last_trans_time
                                            
                                            if time_since_last_transcription >= 2.0:  # At least 2 seconds between transcriptions
                                                logger.info(f"Transcribing: buffer={transcription_buffer_size} bytes, avg_dBFS={avg_dbfs:.2f} dB, silence_ratio={recent_silent_chunks/total_recent_chunks:.2f}, had_speech={had_speech}")
                                                
                                                # Update last transcription time
                                                last_transcription_time[call_sid] = current_time
                                                
                                                # Create a copy of the buffer for transcription
                                                buffer_copy = list(transcription_buffers[call_sid])
                                                
                                                # Clear the transcription buffer to start fresh
                                                transcription_buffers[call_sid] = []
                                                
                                                # Transcribe and respond back to the caller
                                                asyncio.create_task(process_transcription_and_respond(websocket, stream_sid, call_sid, buffer_copy))
                                            else:
                                                logger.debug(f"Skipping transcription (too soon: {time_since_last_transcription:.2f}s since last)")
                                        else:
                                            logger.debug(f"Skipping transcription (no speech detected, only silence)")
                        except Exception as e:
                            logger.error("Failed to decode audio data: %s", e)
                            
                elif event_type == "stop":
                    logger.info("Stream stopped")
                    # Process any remaining transcription buffer and send final response
                    if call_sid and call_sid in transcription_buffers and len(transcription_buffers[call_sid]) > 0:
                        logger.info("Processing final transcription buffer")
                        buffer_copy = list(transcription_buffers[call_sid])
                        asyncio.create_task(process_transcription_and_respond(websocket, stream_sid, call_sid, buffer_copy))
                    
                    # Save the recorded audio when call ends
                    if call_sid and call_sid in call_audio_buffers:
                        audio_buffer = call_audio_buffers[call_sid]
                        logger.info("Call ended. Saving %d bytes of audio for call %s", 
                                  len(audio_buffer), call_sid)
                        
                        # Save as both WAV and raw μ-law
                        save_call_audio(call_sid, audio_buffer)
                        save_raw_mulaw_audio(call_sid, audio_buffer)
                        
                        # Clean up buffers
                        del call_audio_buffers[call_sid]
                        
                    # Clean up transcription buffers
                    if call_sid in transcription_buffers:
                        del transcription_buffers[call_sid]
                    if call_sid in last_audio_time:
                        del last_audio_time[call_sid]
                    if call_sid in decibel_history:
                        del decibel_history[call_sid]
                    if call_sid in last_transcription_time:
                        del last_transcription_time[call_sid]
                    break
                
                message_count += 1
            except json.JSONDecodeError as e:
                logger.error("Failed to parse WebSocket message as JSON: %s", e)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
            except Exception as e:
                logger.error("Error processing WebSocket message: %s", e, exc_info=True)
    finally:
        # Clean up on disconnect
        if call_sid and call_sid in call_audio_buffers:
            del call_audio_buffers[call_sid]
        if call_sid and call_sid in transcription_buffers:
            del transcription_buffers[call_sid]
        if call_sid and call_sid in last_audio_time:
            del last_audio_time[call_sid]
        if call_sid and call_sid in decibel_history:
            del decibel_history[call_sid]
        if call_sid and call_sid in last_transcription_time:
            del last_transcription_time[call_sid]
        logger.info("Media stream connection closed. Processed %d messages", message_count)

async def send_audio_response(websocket: WebSocket, stream_sid: str, text_response: str):
    """Convert text to speech and send as audio"""
    logger.info("🗣️ [TTS] Converting: '%s'", text_response)
    
    try:
        if websocket.client_state.name != "CONNECTED":
            logger.warning("🗣️ [TTS] WebSocket not connected, skipping")
            return
        
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
            
            tts_start_time = time.time()
            response = client.audio.speech.create(
                model="playai-tts",
                voice="Aaliyah-PlayAI",
                input=text_response,
                response_format="wav"
            )
            
            response.write_to_file(tmp_file_path)
            
            # Check sample rate and warn if mismatch
            with wave.open(tmp_file_path, 'rb') as wav_inspect:
                groq_sample_rate = wav_inspect.getframerate()
                if groq_sample_rate != 8000:
                    logger.warning("🗣️ [TTS] Sample rate mismatch: %d Hz → 8000 Hz (quality will degrade)", groq_sample_rate)
            
            with open(tmp_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            if websocket.client_state.name != "CONNECTED":
                return
            
            await send_audio_chunks(websocket, stream_sid, audio_data)
            
            total_time = time.time() - tts_start_time
            logger.info("🗣️ [TTS] Sent in %.2f seconds", total_time)
            
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
            
    except Exception as e:
        logger.error("🗣️ [TTS] Failed: %s", e)
        try:
            if websocket.client_state.name == "CONNECTED":
                await send_silence(websocket, stream_sid)
        except Exception:
            pass

@app.get("/test")
async def test():
    return {
        "message": "Test endpoint working",
        "server": "FastAPI with WebSocket support",
        "port": PORT
    }

@app.get("/recordings", response_class=HTMLResponse)
async def list_recordings():
    """List all recorded audio files with HTML interface"""
    try:
        files = []
        if os.path.exists(AUDIO_STORAGE_DIR):
            for filename in os.listdir(AUDIO_STORAGE_DIR):
                if filename.endswith(('.wav', '.ulaw', '.mp3')):
                    filepath = os.path.join(AUDIO_STORAGE_DIR, filename)
                    file_size = os.path.getsize(filepath)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    files.append({
                        "filename": filename,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024*1024), 2),
                        "modified": file_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "download_url": f"/recordings/{filename}"
                    })
        
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Call Recordings</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .audio-player {{ margin: 10px 0; }}
                .no-recordings {{ text-align: center; color: #666; margin: 50px 0; }}
            </style>
        </head>
        <body>
            <h1>Call Recordings</h1>
            <p>Total recordings: {len(files)}</p>
            
            {"<div class='no-recordings'><h3>No recordings found</h3><p>Make some calls to see recordings here!</p></div>" if not files else ""}
            
            {f'''
            <table>
                <tr>
                    <th>Filename</th>
                    <th>Date/Time</th>
                    <th>Size (MB)</th>
                    <th>Type</th>
                    <th>Actions</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{file["filename"]}</td>
                    <td>{file["modified"]}</td>
                    <td>{file["size_mb"]}</td>
                    <td>{'WAV Audio' if file["filename"].endswith('.wav') else 'MP3 Audio' if file["filename"].endswith('.mp3') else 'μ-law Raw'}</td>
                    <td>
                        <a href="{file["download_url"]}" download>Download</a>
                        {f'<br><audio controls class="audio-player"><source src="{file["download_url"]}" type="audio/wav">Your browser does not support the audio element.</audio>' if file["filename"].endswith('.wav') else f'<br><audio controls class="audio-player"><source src="{file["download_url"]}" type="audio/mpeg">Your browser does not support the audio element.</audio>' if file["filename"].endswith('.mp3') else ''}
                    </td>
                </tr>
                ''' for file in files])}
            </table>
            ''' if files else ""}
            
            <br>
            <p><a href="/">← Back to Home</a></p>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings/{filename}")
async def download_recording(filename: str):
    """Download a specific recording file"""
    try:
        import mimetypes
        
        filepath = os.path.join(AUDIO_STORAGE_DIR, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Security check - ensure file is in the recordings directory
        if not os.path.abspath(filepath).startswith(os.path.abspath(AUDIO_STORAGE_DIR)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Read file content
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Determine media type
        media_type, _ = mimetypes.guess_type(filepath)
        if media_type is None:
            media_type = 'application/octet-stream'
        
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on port %d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT) 