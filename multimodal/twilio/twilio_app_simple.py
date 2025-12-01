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
import shutil
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import logging

# Try to import scipy for high-quality resampling
try:
    import numpy as np
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log scipy availability
if not SCIPY_AVAILABLE:
    logger.warning("scipy/numpy not available, will use audioop for resampling (lower quality)")

# Suppress noisy logs from external libraries
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Configuration
PORT = int(os.getenv('PORT', 5015))

# Initialize Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")
client = Groq(api_key=groq_api_key)

# Audio storage configuration
AUDIO_STORAGE_DIR = "recorded_calls"
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Simplified Twilio Voice Assistant",
    description="Minimal Twilio voice assistant that detects silence, transcribes, and parrots back",
    version="2.0.0"
)

# Global storage per call
audio_buffers = {}  # Single buffer per call
last_audio_time = {}  # Track last audio timestamp
tts_playing = {}  # Track if TTS is currently playing
tts_finished_time = {}  # Track when TTS finished (cooldown period)
had_speech = {}  # Track if we've detected any speech in this buffer
conversation_history = {}  # Track conversation history per call


def mulaw_to_linear(mulaw_data):
    """Convert μ-law encoded audio to linear PCM using audioop (correct implementation)"""
    # audioop.ulaw2lin converts μ-law to 16-bit linear PCM
    # Returns bytes, which we convert to a list of int16 samples
    linear_bytes = audioop.ulaw2lin(mulaw_data, 2)  # 2 = 16-bit samples
    # Convert bytes to list of signed 16-bit integers
    linear_data = list(struct.unpack('<' + 'h' * (len(linear_bytes) // 2), linear_bytes))
    return linear_data


def convert_wav_to_mulaw(wav_file_path):
    """Convert WAV audio to μ-law format for Twilio (simplified, uses audioop only)"""
    try:
        with wave.open(wav_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
            
            # Convert to mono if stereo
            if num_channels == 2:
                frames = audioop.tomono(frames, sample_width, 1, 1)
            
            # Ensure 16-bit samples
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
                logger.info(f"Resampling {sample_rate} Hz → 8000 Hz")
                resampled_result = audioop.ratecv(frames, sample_width, 1, sample_rate, 8000, None)
                frames = resampled_result[0]
            
            # Convert to μ-law (requires 16-bit input)
            mulaw_data = audioop.lin2ulaw(frames, sample_width)
            return mulaw_data
    except Exception as e:
        logger.error(f"Failed to convert WAV to μ-law: {e}")
        return None


async def transcribe_audio(call_sid, audio_buffer):
    """Transcribe audio buffer using Groq Whisper"""
    if not audio_buffer or len(audio_buffer) < 1600:  # At least 100ms
        return None
    
    try:
        logger.info(f"Transcribing audio for call {call_sid} ({len(audio_buffer)} bytes)")
        
        # Convert μ-law to linear PCM
        linear_audio = mulaw_to_linear(audio_buffer)
        
        # Upsample from 8kHz to higher rate for better Whisper quality
        # Whisper works best with 16kHz-48kHz sample rates
        # Higher rates can help even if source is 8kHz (Whisper's internal processing benefits)
        target_sample_rate = 24000  # Try 24kHz - good balance for Whisper
        original_sample_rate = 8000
        
        if SCIPY_AVAILABLE:
            # Use scipy for high-quality upsampling with anti-aliasing
            audio_array = np.array(linear_audio, dtype=np.float32)
            # Normalize to [-1, 1] range for better processing
            audio_array = audio_array / 32768.0
            
            # Remove DC offset (bias) - common in telephone audio
            audio_array = audio_array - np.mean(audio_array)
            
            # Use resample_poly for upsampling (better than resample for integer ratios)
            # resample_poly uses polyphase filtering which is better for upsampling
            upsampled_audio = signal.resample_poly(audio_array, target_sample_rate, original_sample_rate, padtype='line')
            
            # Apply gentler bandpass filter - use Chebyshev type II for smoother response
            nyquist = target_sample_rate / 2
            # High-pass: Remove very low frequencies (rumble/noise)
            high_pass_freq = 80
            # Low-pass: Remove high-frequency artifacts from upsampling (telephone audio is ~300-3400 Hz)
            low_pass_freq = 3400  # Standard telephone bandwidth
            
            # Use Chebyshev type II filters - smoother stopband, less ringing
            sos_high = signal.cheby2(3, 40, high_pass_freq / nyquist, btype='high', output='sos')
            sos_low = signal.cheby2(3, 40, low_pass_freq / nyquist, btype='low', output='sos')
            upsampled_audio = signal.sosfilt(sos_high, upsampled_audio)
            upsampled_audio = signal.sosfilt(sos_low, upsampled_audio)
            
            # RMS normalization (better than peak normalization for speech)
            # This preserves relative dynamics better
            rms = np.sqrt(np.mean(upsampled_audio ** 2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level (adjustable)
                upsampled_audio = upsampled_audio * (target_rms / rms)
                # Clamp to prevent clipping
                upsampled_audio = np.clip(upsampled_audio, -0.95, 0.95)
            
            # Add triangular dithering before quantization to reduce static/quantization noise
            # This helps mask quantization artifacts
            dither = np.random.triangular(-1/32768.0, 0, 1/32768.0, size=len(upsampled_audio))
            upsampled_audio = upsampled_audio + dither
            
            # Convert back to int16
            upsampled_audio = np.clip(upsampled_audio * 32768.0, -32768, 32767).astype(np.int16)
            audio_samples = upsampled_audio.tolist()
            logger.info(f"[{call_sid[:8]}] Upsampled and enhanced audio: 8kHz → {target_sample_rate}Hz (Chebyshev bandpass 80-3400Hz, dithered, RMS normalized)")
        else:
            # Fallback to audioop (lower quality)
            audio_bytes = struct.pack('<' + 'h' * len(linear_audio), *linear_audio)
            upsampled_bytes = audioop.ratecv(audio_bytes, 2, 1, original_sample_rate, target_sample_rate, None)[0]
            audio_samples = list(struct.unpack('<' + 'h' * (len(upsampled_bytes) // 2), upsampled_bytes))
            logger.info(f"[{call_sid[:8]}] Upsampled audio: 8kHz → {target_sample_rate}Hz using audioop")
        
        # Create temporary WAV file with upsampled audio
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                
                with wave.open(tmp_file_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(target_sample_rate)  # 16kHz
                    
                    # Convert to 16-bit signed integers
                    audio_data = struct.pack('<' + 'h' * len(audio_samples), *audio_samples)
                    wav_file.writeframes(audio_data)
            
            # Save a copy to storage directory for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_filename = f"call_{call_sid[:8]}_{timestamp}.wav"
            debug_filepath = os.path.join(AUDIO_STORAGE_DIR, debug_filename)
            shutil.copy2(tmp_file_path, debug_filepath)
            logger.info(f"[{call_sid[:8]}] Saved audio for debugging: {debug_filename}")
            
            # Transcribe with Groq
            logger.info(f"[{call_sid[:8]}] Calling Groq Whisper API...")
            with open(tmp_file_path, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(tmp_file_path, audio_file.read()),
                    model="whisper-large-v3",
                    language="en",  # Specify English for better accuracy
                    prompt="This is a phone call conversation. The speaker is providing their name and zip code.",  # Context helps Whisper
                    response_format="text"
                )
            
            # Clean up temp file (keep the debug copy)
            os.unlink(tmp_file_path)
            
            logger.info(f"[{call_sid[:8]}] Groq API returned: {repr(transcription)}")
            
            if transcription and transcription.strip():
                text = transcription.strip()
                logger.info(f"[{call_sid[:8]}] Transcription result: '{text}' (audio saved as {debug_filename})")
                return text
            else:
                logger.warning(f"[{call_sid[:8]}] Empty transcription returned (audio saved as {debug_filename})")
                return None
            
        except Exception as e:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            raise e
            
    except Exception as e:
        logger.error(f"[{call_sid[:8]}] Transcription failed: {e}", exc_info=True)
        return None


async def get_llm_response(call_sid: str, user_message: str) -> str:
    """Get LLM response using Groq's OpenAI-compatible chat completions API"""
    try:
        # Initialize conversation history if needed
        if call_sid not in conversation_history:
            conversation_history[call_sid] = []
        
        # System prompt for call center agent
        system_prompt = "You are a friendly call center agent trying to gather the caller's name and zip code. Be conversational, natural, and helpful. Keep your responses brief and concise for phone conversations. Start by saying 'Hello, who am I speaking with?'"
        
        # Build messages list
        messages = []
        
        # Add system message if this is the first interaction
        if len(conversation_history[call_sid]) == 0:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history
        messages.extend(conversation_history[call_sid])
        
        # Add current user message if provided
        if user_message:
            messages.append({
                "role": "user",
                "content": user_message
            })
            conversation_history[call_sid].append({
                "role": "user",
                "content": user_message
            })
        else:
            # For initial greeting, add a prompt to start the conversation
            messages.append({"role": "user", "content": "Hello"})
        
        # Call Groq's chat completions API (OpenAI-compatible)
        logger.info(f"[{call_sid[:8]}] Calling Groq LLM API with {len(messages)} messages...")
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=150  # Keep responses brief for phone
        )
        
        # Extract assistant response
        assistant_message = response.choices[0].message.content.strip()
        
        # Add full exchange to history (user message + assistant response)
        if not user_message:
            # This was the initial greeting - add the "Hello" to history
            conversation_history[call_sid].append({
                "role": "user",
                "content": "Hello"
            })
        
        conversation_history[call_sid].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        logger.info(f"[{call_sid[:8]}] LLM response: '{assistant_message}'")
        return assistant_message
        
    except Exception as e:
        logger.error(f"[{call_sid[:8]}] LLM API failed: {e}", exc_info=True)
        # Fallback response
        return "I'm sorry, I'm having trouble processing that. Could you please repeat?"


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
                logger.error("Failed to convert audio to μ-law")
                return
            
            # Send audio in 160-byte chunks (20ms at 8kHz)
            chunk_size = 160
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
                
                if websocket.client_state.name != "CONNECTED":
                    break
                
                await websocket.send_json(audio_message)
                await asyncio.sleep(0.001)  # Send fast to build buffer
            
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Failed to send audio chunks: {e}")


async def send_audio_response(websocket: WebSocket, stream_sid: str, text_response: str, call_sid: str = None):
    """Convert text to speech and send as audio"""
    logger.info(f"[{call_sid[:8] if call_sid else 'NONE'}] TTS: '{text_response}'")
    
    # Mark TTS as playing to prevent feedback loops
    if call_sid:
        tts_playing[call_sid] = True
    
    try:
        if websocket.client_state.name != "CONNECTED":
            logger.warning(f"[{call_sid[:8] if call_sid else 'NONE'}] WebSocket not connected, skipping TTS")
            return
        
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
            
            # Generate TTS
            logger.info(f"[{call_sid[:8] if call_sid else 'NONE'}] Calling Groq TTS API...")
            response = client.audio.speech.create(
                model="playai-tts",
                voice="Aaliyah-PlayAI",
                input=text_response,
                response_format="wav"
            )
            
            response.write_to_file(tmp_file_path)
            logger.info(f"[{call_sid[:8] if call_sid else 'NONE'}] TTS generated, sending audio...")
            
            # Read and send audio
            with open(tmp_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            if websocket.client_state.name == "CONNECTED":
                await send_audio_chunks(websocket, stream_sid, audio_data)
                logger.info(f"[{call_sid[:8] if call_sid else 'NONE'}] TTS audio sent successfully")
            
        finally:
            # Mark TTS as finished
            if call_sid:
                tts_playing[call_sid] = False
                tts_finished_time[call_sid] = time.time()  # Set cooldown timestamp
            
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            
    except Exception as e:
        logger.error(f"[{call_sid[:8] if call_sid else 'NONE'}] TTS failed: {e}", exc_info=True)
        if call_sid:
            tts_playing[call_sid] = False
            tts_finished_time[call_sid] = time.time()


@app.get("/")
async def index():
    return {
        "message": "Simplified Twilio Voice Assistant",
        "status": "running",
        "endpoints": {
            "incoming_call": "/incoming-call",
            "media_stream": "/media-stream"
        }
    }


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response"""
    logger.info("Incoming call received")
    
    response = VoiceResponse()
    
    # Determine protocol (wss for https, ws for http)
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    if forwarded_proto == "https" or request.url.scheme == "https":
        protocol = "wss"
    else:
        protocol = "ws"
    
    host = request.headers.get("host", request.url.hostname)
    
    # Add pause to allow WebSocket connection
    response.pause(length=1)
    
    # Connect to media stream WebSocket
    connect = Connect()
    connect.stream(url=f'{protocol}://{host}/media-stream')
    response.append(connect)
    
    logger.info(f"Returning TwiML with WebSocket URL: {protocol}://{host}/media-stream")
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections for Twilio Media Streams"""
    logger.info("Media stream WebSocket connection initiated")
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket: {e}")
        return
    
    stream_sid = None
    call_sid = None
    websocket_ref = websocket  # Store reference for closure
    
    # Background task to check for silence and transcribe
    async def check_silence_and_transcribe():
        nonlocal call_sid, stream_sid
        check_count = 0
        while True:
            WAIT_TIME = 0.5
            await asyncio.sleep(WAIT_TIME)  # Check every 500ms
            check_count += 1
            
            current_call_sid = call_sid
            current_stream_sid = stream_sid
            
            if not current_call_sid:
                if check_count % 10 == 0:  # Log every 5 seconds
                    logger.debug("Waiting for call to start...")
                continue
            
            # Skip if TTS is playing
            if tts_playing.get(current_call_sid, False):
                continue
            
            # Skip if TTS just finished (2 second cooldown)
            TTS_COOLDOWN_TIME = 2.0
            tts_finished = tts_finished_time.get(current_call_sid, 0)
            if time.time() - tts_finished < TTS_COOLDOWN_TIME:
                continue
            
            # Check if we have audio and haven't received any for 0.5 seconds
            buffer = audio_buffers.get(current_call_sid, [])
            last_time = last_audio_time.get(current_call_sid, 0)
            current_time = time.time()
            time_since_audio = current_time - last_time
            
            # Log buffer status every 2 seconds
            if check_count % (2 / WAIT_TIME) == 0:  # Every 2 seconds (4 * 0.5s)
                time_since_speech = current_time - last_time
                logger.info(f"[{current_call_sid[:8]}] Buffer: {len(buffer)} bytes, time since SPEECH: {time_since_speech:.1f}s, TTS playing: {tts_playing.get(current_call_sid, False)}")
            
            # Transcribe if we have enough audio and silence detected
            # Only transcribe if we've actually had speech (not just silence the whole time)
            if len(buffer) >= 8000 and time_since_audio >= 0.5 and had_speech.get(current_call_sid, False):
                logger.info(f"[{current_call_sid[:8]}] SILENCE DETECTED - Transcribing {len(buffer)} bytes (silence: {time_since_audio:.1f}s)")
                
                # Copy buffer and clear it
                audio_to_transcribe = bytes(buffer)
                audio_buffers[current_call_sid] = []
                had_speech[current_call_sid] = False  # Reset for next utterance
                
                # Transcribe
                transcription = await transcribe_audio(current_call_sid, audio_to_transcribe)
                
                if transcription and len(transcription.strip()) >= 3:
                    logger.info(f"[{current_call_sid[:8]}] Transcription: '{transcription}'")
                    # Get LLM response instead of parroting
                    try:
                        response_text = await get_llm_response(current_call_sid, transcription)
                        await send_audio_response(websocket_ref, current_stream_sid, response_text, current_call_sid)
                    except Exception as e:
                        logger.error(f"[{current_call_sid[:8]}] LLM response failed: {e}", exc_info=True)
                        # Fallback: just parrot back if LLM fails
                        fallback_response = f"I heard you say: {transcription}"
                        await send_audio_response(websocket_ref, current_stream_sid, fallback_response, current_call_sid)
                elif transcription:
                    logger.info(f"[{current_call_sid[:8]}] Ignoring short transcription: '{transcription}'")
                else:
                    logger.warning(f"[{current_call_sid[:8]}] No transcription returned")
    
    # Start background task
    silence_checker = asyncio.create_task(check_silence_and_transcribe())
    
    try:
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                event_type = data.get('event')
                
                if event_type == "connected":
                    logger.info("Media stream connected")
                
                elif event_type == "start":
                    stream_sid = data['start']['streamSid']
                    call_sid = data['start']['callSid']
                    logger.info(f"Stream started - SID: {stream_sid}, Call: {call_sid}")
                    
                    # Initialize buffers
                    audio_buffers[call_sid] = []
                    last_audio_time[call_sid] = time.time()
                    tts_playing[call_sid] = False
                    tts_finished_time[call_sid] = 0
                    had_speech[call_sid] = False
                    conversation_history[call_sid] = []  # Initialize conversation history
                    
                    # Get initial greeting from LLM
                    greeting = await get_llm_response(call_sid, "")
                    await send_audio_response(websocket, stream_sid, greeting, call_sid)
                    tts_finished_time[call_sid] = time.time()  # Set cooldown after greeting
                
                elif event_type == "media":
                    payload = data.get('media', {}).get('payload', '')
                    if payload and call_sid:
                        # Skip if TTS is playing (prevent feedback loop)
                        if tts_playing.get(call_sid, False):
                            continue
                        
                        # Skip if TTS just finished (2 second cooldown)
                        tts_finished = tts_finished_time.get(call_sid, 0)
                        if time.time() - tts_finished < 2.0:
                            continue
                        
                        try:
                            # Decode base64 audio data
                            audio_chunk = base64.b64decode(payload)
                            
                            # Simple VAD: check if this chunk contains speech
                            linear_pcm = audioop.ulaw2lin(audio_chunk, 2)
                            rms = audioop.rms(linear_pcm, 2)
                            
                            # Add all audio to buffer
                            if call_sid not in audio_buffers:
                                audio_buffers[call_sid] = []
                            audio_buffers[call_sid].extend(audio_chunk)
                            
                            # Only update last_audio_time if we detect speech (not silence)
                            # Normal phone speech is around -68 to -74 dB
                            # Silence/background noise is typically < -80 dB
                            # TTS echo might be around -70 to -75 dB
                            if rms > 0:
                                decibels = 20 * math.log10(rms / 32768)
                                # Higher threshold: only consider it speech if > -70 dB (more selective)
                                # This filters out TTS echo and background noise
                                if decibels > -70:
                                    last_audio_time[call_sid] = time.time()
                                    had_speech[call_sid] = True  # Mark that we've had speech
                                    # Log when we detect speech
                                    buffer_size = len(audio_buffers[call_sid])
                                    if buffer_size % 8000 == 0:  # Every ~1 second of audio
                                        logger.info(f"[{call_sid[:8]}] Receiving SPEECH: {buffer_size} bytes, {decibels:.1f} dB")
                            
                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                
                elif event_type == "stop":
                    logger.info("Stream stopped")
                    break
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        silence_checker.cancel()
        if call_sid:
            audio_buffers.pop(call_sid, None)
            last_audio_time.pop(call_sid, None)
            tts_playing.pop(call_sid, None)
            tts_finished_time.pop(call_sid, None)
            had_speech.pop(call_sid, None)
            conversation_history.pop(call_sid, None)  # Clean up conversation history
        logger.info("Media stream connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

