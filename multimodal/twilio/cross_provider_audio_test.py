import os
import time
import tempfile
import wave
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossProviderAudioTester:
    def __init__(self):
        self.results = []
        
        # Simplified test set
        self.test_phrases = [
            "Hello, this is a test of the speech recognition system.",
            "The quick brown fox jumps over the lazy dog.", 
            "Can you please transcribe this audio accurately?"
        ]
        
        # Standard 8kHz sample rate for all tests (phone calls are 8kHz)
        self.sample_rate = 8000
        
        # Initialize clients
        self.openai_client = None
        self.deepgram_client = None
        self.groq_client = None
        
        self._initialize_clients()
        
        # Model configurations
        self.openai_tts_models = ["tts-1", "gpt-4o-mini-tts", "tts-1-hd"]
        self.openai_tts_voices = ["alloy", "nova"]
        self.openai_stt_models = ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
        
        self.deepgram_tts_models = ["aura-2-athena-en", "aura-2-luna-en"]
        self.deepgram_stt_models = ["nova-3", "nova-2"]
        
        self.groq_tts_models = ["playai-tts"]
        self.groq_tts_voices = ["Aaliyah-PlayAI"]
        self.groq_stt_models = ["whisper-large-v3-turbo", "distil-whisper-large-v3-en", "whisper-large-v3"]

    def _initialize_clients(self):
        """Initialize OpenAI, Deepgram, and Groq clients"""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI()
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            
        try:
            from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions
            api_key = os.getenv('DEEPGRAM_API_KEY')
            if not api_key:
                raise ValueError("DEEPGRAM_API_KEY environment variable not set")
            self.deepgram_client = DeepgramClient(api_key)
            logger.info("✅ Deepgram client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Deepgram: {e}")
            
        try:
            from groq import Groq
            self.groq_client = Groq()
            logger.info("✅ Groq client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq: {e}")

    def create_test_audio(self, text: str, frequency: int = 440, duration: float = 1.0) -> str:
        """Create a simple test audio file"""
        import math
        import struct
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Create a simple sine wave
            frames = []
            for i in range(int(self.sample_rate * duration)):
                value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / self.sample_rate))
                frames.append(struct.pack('<h', value))
            
            # Write WAV file
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(frames))
            
            return tmp_file.name

    def openai_tts(self, text: str, model: str, voice: str) -> tuple:
        """Generate speech using OpenAI TTS"""
        try:
            start_time = time.time()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                response = self.openai_client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=text,
                    response_format="wav"
                )
                
                # Write to file
                response.write_to_file(tmp_file.name)
                
                generation_time = time.time() - start_time
                return tmp_file.name, generation_time
                
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return None, None

    def openai_stt(self, audio_file: str, model: str) -> tuple:
        """Transcribe audio using OpenAI STT"""
        try:
            start_time = time.time()
            
            with open(audio_file, 'rb') as audio:
                response = self.openai_client.audio.transcriptions.create(
                    model=model,
                    file=audio,
                    response_format="text"
                )
                
                transcription_time = time.time() - start_time
                return response, transcription_time
                
        except Exception as e:
            logger.error(f"OpenAI STT error: {e}")
            return None, None

    def deepgram_tts(self, text: str, model: str) -> tuple:
        """Generate speech using Deepgram TTS"""
        try:
            from deepgram import SpeakOptions
            
            start_time = time.time()
            
            options = SpeakOptions(
                model=model,
                encoding="linear16",
                sample_rate=self.sample_rate
            )
            
            response = self.deepgram_client.speak.v("1").save(
                filename="temp_deepgram_audio.wav",
                source={"text": text},
                options=options
            )
            
            generation_time = time.time() - start_time
            return "temp_deepgram_audio.wav", generation_time
            
        except Exception as e:
            logger.error(f"Deepgram TTS error: {e}")
            return None, None

    def deepgram_stt(self, audio_file: str, model: str) -> tuple:
        """Transcribe audio using Deepgram STT"""
        try:
            from deepgram import PrerecordedOptions
            
            start_time = time.time()
            
            options = PrerecordedOptions(
                model=model,
                language="en-US",
                smart_format=True,
                punctuate=True
            )
            
            with open(audio_file, 'rb') as audio:
                buffer_data = audio.read()
                
            payload = {"buffer": buffer_data}
            response = self.deepgram_client.listen.prerecorded.v("1").transcribe_file(
                payload,
                options
            )
            
            transcription_time = time.time() - start_time
            
            # Extract text
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript, transcription_time
            
        except Exception as e:
            logger.error(f"Deepgram STT error: {e}")
            return None, None

    def groq_tts(self, text: str, model: str, voice: str) -> tuple:
        """Generate speech using Groq TTS"""
        try:
            start_time = time.time()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                response = self.groq_client.audio.speech.create(
                    model=model,
                    voice=voice,
                    response_format="wav",
                    input=text
                )
                
                # Use the correct method name from Groq API
                response.write_to_file(tmp_file.name)
                generation_time = time.time() - start_time
                return tmp_file.name, generation_time
                
        except Exception as e:
            logger.error(f"Groq TTS error: {e}")
            return None, None

    def groq_stt(self, audio_file: str, model: str) -> tuple:
        """Transcribe audio using Groq STT"""
        try:
            start_time = time.time()
            
            with open(audio_file, 'rb') as audio:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(audio_file, audio.read()),
                    model=model,
                    response_format="text"
                )
                
                transcription_time = time.time() - start_time
                return transcription, transcription_time
                
        except Exception as e:
            logger.error(f"Groq STT error: {e}")
            return None, None

    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate between reference and hypothesis"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Simple WER calculation
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
            
        # Count differences (simplified)
        max_len = max(len(ref_words), len(hyp_words))
        errors = 0
        
        for i in range(max_len):
            ref_word = ref_words[i] if i < len(ref_words) else ""
            hyp_word = hyp_words[i] if i < len(hyp_words) else ""
            if ref_word != hyp_word:
                errors += 1
                
        return errors / len(ref_words)

    def run_cross_provider_roundtrip(self, text: str, 
                                   tts_provider: str, tts_model: str, tts_voice: str,
                                   stt_provider: str, stt_model: str) -> Dict[str, Any]:
        """Run a complete roundtrip test between providers"""
        
        logger.info(f"🔄 Roundtrip: {tts_provider} TTS → {stt_provider} STT")
        logger.info(f"   Text: '{text[:50]}...'")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'tts_provider': tts_provider,
            'tts_model': tts_model,
            'tts_voice': tts_voice,
            'stt_provider': stt_provider, 
            'stt_model': stt_model,
            'roundtrip_type': f"{tts_provider}_TTS → {stt_provider}_STT"
        }
        
        # Step 1: Generate audio
        if tts_provider == "openai":
            audio_file, tts_time = self.openai_tts(text, tts_model, tts_voice)
        elif tts_provider == "deepgram":
            audio_file, tts_time = self.deepgram_tts(text, tts_model)
        elif tts_provider == "groq":
            audio_file, tts_time = self.groq_tts(text, tts_model, tts_voice)
        else:
            return None
            
        if not audio_file:
            result.update({'status': 'failed', 'error': 'TTS generation failed'})
            return result
            
        result['tts_time'] = tts_time
        
        # Step 2: Transcribe audio
        if stt_provider == "openai":
            transcribed_text, stt_time = self.openai_stt(audio_file, stt_model)
        elif stt_provider == "deepgram":
            transcribed_text, stt_time = self.deepgram_stt(audio_file, stt_model)
        elif stt_provider == "groq":
            transcribed_text, stt_time = self.groq_stt(audio_file, stt_model)
        else:
            return None
            
        if transcribed_text is None:
            result.update({'status': 'failed', 'error': 'STT transcription failed'})
            return result
            
        result['stt_time'] = stt_time
        result['transcribed_text'] = transcribed_text
        result['total_time'] = tts_time + stt_time
        
        # Calculate accuracy metrics
        wer = self.calculate_word_error_rate(text, transcribed_text)
        accuracy = max(0, 1 - wer) * 100
        
        result.update({
            'word_error_rate': wer,
            'accuracy_percentage': accuracy,
            'status': 'success'
        })
        
        # Cleanup
        try:
            os.unlink(audio_file)
        except:
            pass
            
        logger.info(f"   ✅ Accuracy: {accuracy:.1f}% | Total Time: {result['total_time']:.2f}s")
        
        return result

    def run_comprehensive_test(self):
        """Run comprehensive cross-provider testing"""
        logger.info("🚀 Starting Cross-Provider Audio Testing (OpenAI + Deepgram + Groq)")
        logger.info(f"📝 Testing {len(self.test_phrases)} phrases at {self.sample_rate}Hz")
        
        all_results = []
        
        # Define test combinations (tts -> stt)
        test_combinations = [
            # Same-provider baselines
            ("openai", "openai"),
            ("deepgram", "deepgram"), 
            ("groq", "groq"),
            # Cross-provider combinations
            ("openai", "deepgram"),
            ("openai", "groq"),
            ("deepgram", "openai"),
            ("deepgram", "groq"),
            ("groq", "openai"),
            ("groq", "deepgram")
        ]
        
        for phrase_idx, phrase in enumerate(self.test_phrases):
            logger.info(f"\n📝 Testing phrase {phrase_idx + 1}/{len(self.test_phrases)}")
            
            for tts_provider, stt_provider in test_combinations:
                
                # Get model combinations for this provider pair
                if tts_provider == "openai":
                    tts_models = [(model, voice) for model in self.openai_tts_models 
                                                 for voice in self.openai_tts_voices]
                elif tts_provider == "deepgram":
                    tts_models = [(model, None) for model in self.deepgram_tts_models]
                elif tts_provider == "groq":
                    tts_models = [(model, voice) for model in self.groq_tts_models 
                                                 for voice in self.groq_tts_voices]
                else:
                    continue
                    
                if stt_provider == "openai":
                    stt_models = self.openai_stt_models
                elif stt_provider == "deepgram":
                    stt_models = self.deepgram_stt_models
                elif stt_provider == "groq":
                    stt_models = self.groq_stt_models
                else:
                    continue
                
                # Test ALL model combinations for each provider pair
                for tts_model, tts_voice in tts_models:
                    for stt_model in stt_models:
                        result = self.run_cross_provider_roundtrip(
                            phrase, tts_provider, tts_model, tts_voice or "N/A", 
                            stt_provider, stt_model
                        )
                        if result:
                            all_results.append(result)
                    
                time.sleep(0.2)  # Brief pause between requests
        
        # Save results
        self.save_results(all_results)
        self.display_summary(all_results)
        
        return all_results

    def save_results(self, results: List[Dict[str, Any]]):
        """Save test results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON
        json_filename = f"cross_provider_test_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {json_filename}")
        
        # Save to CSV
        try:
            df = pd.DataFrame(results)
            csv_filename = f"cross_provider_test_results_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            logger.info(f"💾 Results saved to {csv_filename}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def display_summary(self, results: List[Dict[str, Any]]):
        """Display test summary"""
        logger.info("\n" + "="*80)
        logger.info("🏁 CROSS-PROVIDER AUDIO TEST SUMMARY")
        logger.info("="*80)
        
        # Group by roundtrip type
        roundtrip_stats = {}
        
        for result in results:
            if result['status'] != 'success':
                continue
                
            rt_type = result['roundtrip_type']
            if rt_type not in roundtrip_stats:
                roundtrip_stats[rt_type] = {
                    'times': [],
                    'accuracies': [],
                    'count': 0
                }
                
            roundtrip_stats[rt_type]['times'].append(result['total_time'])
            roundtrip_stats[rt_type]['accuracies'].append(result['accuracy_percentage'])
            roundtrip_stats[rt_type]['count'] += 1
        
        # Display stats
        for rt_type, stats in roundtrip_stats.items():
            if stats['count'] == 0:
                continue
                
            avg_time = sum(stats['times']) / len(stats['times'])
            avg_accuracy = sum(stats['accuracies']) / len(stats['accuracies'])
            
            logger.info(f"\n🔄 {rt_type}:")
            logger.info(f"   ⏱️  Average Time: {avg_time:.2f}s")
            logger.info(f"   🎯 Average Accuracy: {avg_accuracy:.1f}%")
            logger.info(f"   📊 Tests Completed: {stats['count']}")

def main():
    """Main execution function"""
    # Check dependencies
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        os.system("pip install pandas")
        import pandas as pd
    
    tester = CrossProviderAudioTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 