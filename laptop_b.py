import sounddevice as sd
import socket
import numpy as np
import threading
import queue
import time
import sys
try:
    from pipeline import VortexPipeline
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False

# Network Settings
PORT = 5005
SAMPLE_RATE = 16000
CHANNELS = 1

# Buffers
audio_queue = queue.Queue(maxsize=100)
analysis_buffer = bytearray()

shutdown_flag = threading.Event()

def detect_watermark(audio_array, sample_rate):
    """
    Calculates the FFT to see if the 7.5kHz watermark tone survived the round trip.
    If it is missing, a neural vocoder (Voice Clone) likely stripped it.
    """
    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(audio_array))
    fft_freqs = np.fft.rfftfreq(len(audio_array), 1.0/sample_rate)
    
    # Look for energy near 7500 Hz
    target_freq = 7500.0
    tolerance = 100.0
    
    # Indices corresponding to the frequency band
    band_indices = np.where((fft_freqs > target_freq - tolerance) & (fft_freqs < target_freq + tolerance))[0]
    
    if len(band_indices) == 0:
        return False
        
    watermark_energy = np.max(fft_vals[band_indices])
    baseline_energy = np.mean(fft_vals)
    
    # If the watermark peak is significantly higher than the baseline noise floor, it survived
    if watermark_energy > baseline_energy * 5.0:
        return True
    return False

def audio_playback_worker():
    def callback(outdata, frames, time_info, status):
        try:
            data = audio_queue.get_nowait()
            outdata[:, 0] = data
        except queue.Empty:
            outdata.fill(0)
            
        rms = np.sqrt(np.mean(outdata**2))
        sys.stdout.write(f"\r[Audio Level: {rms:.4f}] Playback Queue: {len(audio_queue.queue)} blocks.  ")
        sys.stdout.flush()
            
    try:
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                             dtype='float32', blocksize=1024, 
                             callback=callback):
            print("\n🔊 Speaker Playback Active")
            while not shutdown_flag.is_set():
                sd.sleep(100)
    except Exception as e:
        print(f"Playback error: {e}")

def analysis_worker():
    global analysis_buffer
    
    if HAS_PIPELINE:
        print("\n🤖 Initializing VORTEX Next-Gen AI Engine (Wav2Vec2 + HuBERT + KAN)...")
        engine = VortexPipeline()
        print("✅ VORTEX AI Engine Ready.")
    else:
        print("⚠️ VORTEX Models not found. Running in MOCK mode.")
        engine = None
        
    chunk_requirement = SAMPLE_RATE * 4 * 3 # Float32 (4 bytes). 3 seconds.

    while not shutdown_flag.is_set():
        if len(analysis_buffer) >= chunk_requirement:
            chunk_bytes = analysis_buffer[:chunk_requirement]
            del analysis_buffer[:chunk_requirement]
            
            # Convert to float32 numpy array
            float_array = np.frombuffer(chunk_bytes, dtype=np.float32)
            
            print(f"\n\n🔍 Analyzing 3-Second Chunk...")
            
            # --- 1. Watermark Detection (Honeypot) ---
            watermark_survived = detect_watermark(float_array, SAMPLE_RATE)
            if not watermark_survived:
                print("⚠️  [WATERMARK MISSING] Honeypot tone was stripped! Potential Vocoder detected.")
            else:
                print("🛡️  [WATERMARK INTACT] Mathematical honeypot survived.")
                
            # --- 2. VORTEX Pipeline Analysis ---
            if engine:
                try:
                    # The pipeline now accepts raw 16-bit PCM bytes for the websocket integration normally,
                    # but since we have a float array in the laptop script:
                    # We will format it exactly as `pipeline.process_chunk` expects: 16-bit PCM bytes.
                    pcm_array = (float_array * 32768.0).clip(-32768, 32767).astype(np.int16)
                    pcm_bytes = pcm_array.tobytes()
                    
                    telemetry = engine.process_chunk(pcm_bytes, SAMPLE_RATE)
                    
                    print(f"🧠 Fusion Score: {telemetry['fusion_score']:.4f} "
                          f"(Neural: {telemetry['layers']['neural']:.2f}, "
                          f"Heuristic: {telemetry['layers']['heuristic']:.2f}, "
                          f"Manifold: {telemetry['layers']['manifold']:.2f})")
                    print(f"⚖️  Agreements: {telemetry['layers']['agreements']}/3 Layers")
                    print(f"🎙️  ASR: \"{telemetry['transcript']}\"")
                    
                    verdict = telemetry['verdict']
                    print(f"🎯 Verdict: {verdict}")
                    
                    if verdict == "FAKE" or (verdict == "SUSPICIOUS" and not watermark_survived):
                        print("\n🚨 [VOICE CLONE DETECTED] 🚨")
                        print("📡 TERMINATING CONNECTION TO PROTECT USER...")
                        shutdown_flag.set()
                        break
                        
                except Exception as e:
                    print(f"Analysis engine error: {e}")
            else:
                time.sleep(0.5)
                
        time.sleep(0.1)

def start_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    
    print(f"🛡️ VORTEX Target Receiver & Analyzer")
    print(f"👂 Listening on UDP Port {PORT}...")
    
    playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
    playback_thread.start()
    
    analysis_t = threading.Thread(target=analysis_worker, daemon=True)
    analysis_t.start()
    
    global analysis_buffer
    
    try:
        while not shutdown_flag.is_set():
            data, addr = sock.recvfrom(65536)
            if shutdown_flag.is_set():
                break

            float_array = np.frombuffer(data, dtype=np.float32)
            
            if not audio_queue.full():
                audio_queue.put_nowait(float_array)
                
            analysis_buffer.extend(data)
            
    except KeyboardInterrupt:
        print("\n⏹️ Receiver Stopped Manually.")
        shutdown_flag.set()
    finally:
        sock.close()
        sys.exit(0)

if __name__ == "__main__":
    start_receiver()
