import sounddevice as sd
import socket
import argparse
import numpy as np

# Network Settings
PORT = 5005
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024  # Small block size for low latency

# -------------------------------------------------------------
# Active Defense Generators
# -------------------------------------------------------------

def generate_watermark(block_size, sample_rate, t_start):
    """
    1. Frequency Watermark Honeypot
    Plays a faint, high-frequency sweeping sine wave tone (e.g., 7.5kHz).
    If the remote end uses a neural vocoder, this will be destroyed.
    """
    t = t_start + np.arange(block_size) / sample_rate
    # 7500 Hz faint sine wave
    watermark_freq = 7500.0 
    watermark = 0.05 * np.sin(2 * np.pi * watermark_freq * t)
    return watermark.astype(np.float32)

def generate_fuzzing_noise(block_size):
    """
    2. Acoustic Fuzzing
    Injects high-frequency/structural noise. 
    16kHz Nyquist limit is 8kHz, so we inject near 7.9kHz to target the upper 
    bounds of typical telephony codecs or real-time TTS/RVC encoders.
    We'll produce rapid, aggressive phase shifts.
    """
    # Generate random high-frequency structural noise
    noise = np.random.normal(0, 0.05, block_size).astype(np.float32)
    # High-pass filter approximation (simple differencing)
    noise[1:] = noise[1:] - noise[:-1]
    return noise

def generate_adversarial_pattern(block_size):
    """
    3. Acoustic Adversarial Example (Simplified PGD/FGSM attack simulation)
    Injects a patterned structural noise designed to trigger Whisper hallucinations.
    (In a real production system, this would be a pre-computed PGD attack audio file.
    Here we simulate the concept with a specific broadband pulse train).
    """
    # A pulse train can heavily confuse ASR models if placed at specific intervals
    pattern = np.zeros(block_size, dtype=np.float32)
    pattern[::50] = 0.15 # Pulse every 50 samples
    return pattern

# -------------------------------------------------------------

def start_streaming(receiver_ip, enable_fuzzing, enable_watermark, enable_adv):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (receiver_ip, PORT)
    
    print(f"🎤 VORTEX Transmitter Active")
    print(f"📡 Streaming to {receiver_ip}:{PORT}")
    print(f"⚙️  {SAMPLE_RATE}Hz, Mono, Float32, Block Size: {BLOCK_SIZE}")
    print(f"🛡️  Active Defenses: Fuzzing={enable_fuzzing}, Watermark={enable_watermark}, Adversarial={enable_adv}")
    
    try:
        device_info = sd.query_devices(sd.default.device[0]) if sd.default.device[0] is not None else None
        mic_name = device_info['name'] if device_info else "Unknown"
        print(f"🎙 Active Microphone: {mic_name}")
    except Exception as e:
        print(f"⚠️ Warning, could not query active microphone: {e}")
        
    print("Press Ctrl+C to stop streaming...")

    # Keep track of time for continuous waveforms (like the watermark)
    time_tracker = {"t": 0.0}

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass # Ignore standard underflow prints for cleaner console
            
        # 1. Base Mic Audio
        # Amplify signal natively
        audio_frame = indata[:, 0].astype(np.float32) * 5.0
        
        # 2. Inject Active Defenses (The Honeypot / Attack)
        if enable_watermark:
            wm = generate_watermark(frames, SAMPLE_RATE, time_tracker["t"])
            audio_frame += wm
            
        if enable_fuzzing:
            fz = generate_fuzzing_noise(frames)
            audio_frame += fz
            
        if enable_adv:
            adv = generate_adversarial_pattern(frames)
            audio_frame += adv
            
        time_tracker["t"] += frames / SAMPLE_RATE

        # Prevent clipping after injection
        audio_frame = np.clip(audio_frame, -1.0, 1.0)
        
        # Convert float32 numpy array to raw bytes
        raw_bytes = audio_frame.tobytes()
        
        # Send via UDP
        try:
            sock.sendto(raw_bytes, server_address)
        except Exception as e:
            print(f"Network error: {e}")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                            dtype='float32', blocksize=BLOCK_SIZE, 
                            callback=audio_callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n⏹️ Streaming stopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laptop A: VORTEX Microphone Sender with Active Defenses")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target IP")
    parser.add_argument("--fuzz", action="store_true", help="Enable Acoustic Fuzzing to crash TTS")
    parser.add_argument("--watermark", action="store_true", help="Enable Frequency Watermark Honeypot")
    parser.add_argument("--adv", action="store_true", help="Enable Adversarial Pattern for Whisper")
    args = parser.parse_args()
    
    start_streaming(args.ip, args.fuzz, args.watermark, args.adv)
