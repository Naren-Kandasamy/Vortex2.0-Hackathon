import os
import io
import torch
import torchaudio
import numpy as np
import pickle
from transformers import pipeline
from models import HybridVortexModel

class VortexPipeline:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing VORTEX Next-Gen Pipeline on {self.device}...")
        
        # 1. Hybrid Architecture (Heavy-Light)
        self.hybrid_model = HybridVortexModel(device=self.device).to(self.device)
        self.hybrid_model.eval()
        
        # Load trained core weights
        try:
            self.hybrid_model.core.load_state_dict(torch.load('wavespnet_core.pth', map_location=self.device))
            print("Loaded wavespnet_core.pth successfully.")
        except FileNotFoundError:
            print("Warning: wavespnet_core.pth not found. Using untrained acoustic weights for MVP showcase.")

        # Load Manifold GMM (for S_Manifold)
        self.gmm = None
        if os.path.exists('wavespnet_gmm.pkl'):
            with open('wavespnet_gmm.pkl', 'rb') as f:
                self.gmm = pickle.load(f)
            print("Loaded wavespnet_gmm.pkl successfully.")
        else:
            print("Warning: wavespnet_gmm.pkl not found. S_Manifold will default to 0.0.")

        # 2. Linguistic Intent Analysis
        print("Loading Whisper ASR (tiny.en)...")
        self.asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=0 if self.device=='cuda' else -1)
        
        print("Loading DeBERTa Zero-Shot Intent Classifier...")
        self.intent_classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small", device=0 if self.device=='cuda' else -1)
        
        self.scam_labels = ["urgency", "authority impersonation", "financial demand"]

    def engineered_heuristic_scores(self, waveform):
        """
        Calculates S_Heuristic based on Micro-Timer Jitter and Spectral Entropy.
        waveform: 1D numpy array of shape [48000]
        """
        # 1. Micro-Timer Jitter (Band-pass logic)
        # Calculates peak-to-peak energy stability.
        # AI is often too smooth, or abruptly noisy due to artifacting.
        frame_size = 160
        num_frames = len(waveform) // frame_size
        energies = []
        for i in range(num_frames):
            frame = waveform[i*frame_size:(i+1)*frame_size]
            energies.append(np.sum(frame**2))
            
        energies = np.array(energies)
        jitter = np.mean(np.abs(np.diff(energies))) # Stability measure
        
        # Band-pass: if jitter is extremely low (too smooth TTS) or extremely high (vocoder glitching)
        if jitter < 0.0005 or jitter > 1.0: # Relaxed tolerance
            jitter_penalty = 1.0
        else:
            jitter_penalty = 0.0
            
        # 2. Spectral Entropy (Irregularity of the signal)
        stft = np.abs(torchaudio.functional.spectrogram(torch.from_numpy(waveform).unsqueeze(0), pad=0, window=torch.hann_window(400), n_fft=512, hop_length=160, win_length=400, power=2.0, normalized=False)).numpy().squeeze(0)
        # Normalize sum across freqs per time frame
        prob_dist = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10), axis=0) # [time]
        mean_entropy = np.mean(entropy)
        
        # Human speech is spiky (lower localized entropy). Flat entropy = AI sampled.
        # Relaxed entropy penalty curve
        entropy_score = min(max((mean_entropy - 3.5) / 5.0, 0.0), 1.0)
        
        heuristic_score = 0.5 * jitter_penalty + 0.5 * entropy_score
        return float(heuristic_score)


    @torch.no_grad()
    def process_chunk(self, audio_chunk_bytes, sample_rate=16000):
        # 1. Audio Ingestion & Preprocessing
        # Convert raw bytes (16-bit PCM) to Float Tensor, Peak Normalize to 1.0
        audio_array = np.frombuffer(audio_chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Peak normalization
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
            
        waveform = torch.from_numpy(audio_array).to(torch.float32)
        
        # Pad or truncate to 48,000 samples
        target_size = 48000
        if waveform.shape[0] > target_size:
            acoustic_input = waveform[:target_size]
        else:
            padding = target_size - waveform.shape[0]
            acoustic_input = torch.nn.functional.pad(waveform, (0, padding))
            
        acoustic_input = acoustic_input.unsqueeze(0).to(self.device) # [1, 48000]
        
        # 2. Stage 7 Fusion Formula (S_final = 0.5*Neural + 0.2*Heuristic + 0.3*Manifold)
        # A) Neural Score (S_Neural) via Hybrid Stack
        neural_prob, hubert_emb = self.hybrid_model(acoustic_input)
        s_neural = neural_prob.item()
        
        # B) Heuristic Score (S_Heuristic)
        s_heuristic = self.engineered_heuristic_scores(acoustic_input.squeeze(0).cpu().numpy())
        
        # C) Manifold Score (S_Manifold) via GMM Anomaly Detection
        if self.gmm is not None:
            emb_np = hubert_emb.cpu().numpy()
            # Score samples returns log probability. Lower = anomalous (Fake). higher = Human
            log_prob = self.gmm.score_samples(emb_np)[0]
            # Convert log prob to an anomaly score 0-1.
            # Typical log_probs might range from heavily negative to positive.
            # A completely empirical threshold mapping for MVP:
            # We shift the sigmoid center to make it less aggressive at flagging normal speech
            s_manifold = 1.0 / (1.0 + np.exp(log_prob + 80.0)) # Maps large neg to 1 (Fake), large pos to 0 (Real)
        else:
            s_manifold = 0.0
            
        s_final = 0.50 * s_neural + 0.20 * s_heuristic + 0.30 * s_manifold
        
        # 3. Final Multi-Gate Verdict
        # If 3/3 layers agree (>0.5) -> FAKE
        # If 2/3 agree -> SUSPICIOUS (Changed from 1/3 to reduce false positives on mic noise)
        agreements = sum([s_neural > 0.5, s_heuristic > 0.5, s_manifold > 0.5])
        
        if agreements == 3:
            verdict = "FAKE"
        elif agreements == 2:
            verdict = "SUSPICIOUS"
        else:
            verdict = "REAL"
            
        # Due to hackathon MVP safety, optionally override to REAL if not perfectly certain to keep stream alive
        # For full demo logic we return the real verdict via telemetry regardless.
        
        # 4. ASR & Intent
        transcript = ""
        try:
            asr_out = self.asr_pipeline({"raw": acoustic_input.squeeze(0).cpu().numpy(), "sampling_rate": sample_rate})
            transcript = asr_out.get("text", "").strip()
        except Exception as e:
            pass
            
        intent_scores = {}
        max_intent_confidence = 0.0
        if len(transcript.split()) > 2:
            try:
                intent_out = self.intent_classifier(transcript, candidate_labels=self.scam_labels)
                labels = intent_out['labels']
                scores = intent_out['scores']
                intent_scores = {labels[i]: round(scores[i], 3) for i in range(len(labels))}
                max_intent_confidence = max(scores)
            except Exception as e:
                pass
        else:
            intent_scores = {lbl: 0.0 for lbl in self.scam_labels}
            
        return {
            "verdict": verdict,
            "fusion_score": round(s_final, 4),
            "layers": {
                "neural": round(s_neural, 4),
                "heuristic": round(s_heuristic, 4),
                "manifold": round(s_manifold, 4),
                "agreements": agreements
            },
            "transcript": transcript,
            "intent_confidence": round(max_intent_confidence, 4),
            "intent_breakdown": intent_scores
        }
