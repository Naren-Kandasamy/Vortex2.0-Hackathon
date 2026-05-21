# Vortex 2.0: AI Voice Fraud Detection System

A cutting-edge **real-time voice fraud detection system** powered by advanced deep learning and acoustic analysis. Vortex uses a novel **hybrid multi-stage architecture** combining neural networks, signal processing, and statistical manifold analysis to detect synthetic, deepfake, and spoofed audio with high accuracy.

[![Language](https://img.shields.io/badge/Python-70.5%25-blue)](https://python.org)
[![Frontend](https://img.shields.io/badge/HTML-29.5%25-orange)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![Framework](https://img.shields.io/badge/FastAPI-WebSocket-brightgreen)](#)
[![Status](https://img.shields.io/badge/Status-Hackathon%20MVP-yellow)](#)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Innovation](#-key-innovation)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Model Components](#-model-components)
- [Training](#-training)
- [Configuration](#-configuration)
- [WebSocket API](#-websocket-api)
- [Performance Metrics](#-performance-metrics)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

**Vortex 2.0** is an intelligent defense system against voice fraud and deepfake audio attacks. In an era where AI-generated voice can be nearly indistinguishable from real human speech, Vortex provides real-time detection capabilities.

### Problem Statement

- **Voice fraud** is rapidly escalating as TTS (Text-to-Speech) and voice cloning technologies become more accessible
- **Deepfake audio** can impersonate trusted individuals in sensitive scenarios (corporate fraud, financial crimes, social engineering)
- Existing detection methods are often **slow, require offline processing**, or are easily defeated by sophisticated synthesizers
- There's a critical need for **real-time, multi-layer detection** that combines acoustic analysis with linguistic intent recognition

### Solution

Vortex employs a **seven-stage fusion pipeline**:

1. **Neural Analysis** - WaveSpNetCore (hybrid deep learning model)
2. **Heuristic Signals** - Micro-timer jitter and spectral entropy analysis
3. **Manifold Anomaly Detection** - Gaussian Mixture Model on speech embeddings
4. **Linguistic Intent Analysis** - ASR + zero-shot classification for scam detection
5. **Multi-Gate Voting** - Consensus verdict from all layers
6. **Real-time Processing** - WebSocket-based streaming pipeline
7. **Interactive Dashboard** - Live verdict feedback UI

---

## 🚀 Key Innovation

### The Hybrid Architecture (Heavy-Light)

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Audio (48kHz, 3-sec)               │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴───────────────────┐
        │                                    │
    ┌───▼────────┐              ┌──────────▼──────────┐
    │  Wav2Vec2  │              │      HuBERT        │
    │ (Frozen)   │              │     (Frozen)       │
    └───┬────────┘              └──────────┬──────────┘
        │                                  │
        └────────────────┬─────────────────┘
                         │
        ┌────────────────┴────────────────────┐
        │                                     │
    ┌───▼──────────┐            ┌────────────▼────────────┐
    │ LWT Features │            │  NCRA (Neural Codec)   │
    │   (256-dim)  │            │     (256-dim)          │
    └───┬──────────┘            └────────────┬────────────┘
        │                                    │
        └─────────────┬──────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   Conformer Fusion Block   │
        │  (2 Conformer Blocks)      │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   GRU Backend (2-layer)    │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │  Exact Spline KAN Head     │
        │  (Kolmogorov-Arnold Network)
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   Fake Probability [0, 1]  │
        │   + HuBERT Embeddings      │
        └────────────────────────────┘
```

**Key Features:**
- **Frozen Foundations** (Wav2Vec2, HuBERT) leverage pre-trained acoustic knowledge
- **Learnable Components** (LWT, NCRA, Conformer, GRU, KAN) are optimized for fake detection
- **Memory Efficient** - Frozen models reduce parameters; only ~2M trainable params in core
- **State-of-the-art Architecture** - Combines wavelets, neural codecs, transformers, and KAN layers

---

## 🏗 Architecture

### Five-Stage Processing Pipeline

#### **Stage 1: Frozen Foundation Backbones**
- **Wav2Vec2** (facebook/wav2vec2-base-960h) - 768-dim acoustic representations
- **HuBERT** (facebook/hubert-base-ls960) - 768-dim linguistic representations
- Both frozen to preserve pre-trained knowledge

#### **Stage 2: Feature Extraction**

**Learnable Wavelet Transform (LWT)**
- Decomposes audio into 8 subbands using learnable filters
- Applies temporal CNN (8 → 64 → 128 → 256 channels)
- Output: [B, 256, 149] feature maps

**Neural Codec Residual Analyzer (NCRA)**
- Processes STFT magnitude spectrograms (513 frequency bins)
- 4-layer CNN → GRU backend
- Output: [B, 149, 256] residual analysis features

#### **Stage 3: Conformer Fusion**
- Concatenates: Wav2Vec2 [768] + NCRA [256] + LWT [256] = 1280 dims
- Pads to 1352 and projects to 512-dim embedding space
- 2 Conformer blocks with Multi-Head Attention (8 heads), depthwise convolutions, and feed-forward layers
- Output: [B, 149, 256] sequence

#### **Stage 4: GRU Backend**
- 2-layer bidirectional GRU (input_size=256, hidden_size=256)
- Extracts temporal dependencies and final state representation
- Output: [B, 256] hidden state + scalar auxiliary logit

#### **Stage 5: Exact Spline KAN Head**
- Kolmogorov-Arnold Network with learnable basis functions
- Layer 1: 256 → 16 (spline KAN)
- Layer 2: 16 → 1 (spline KAN)
- Output: [0, 1] probability via sigmoid

### Three-Layer Fusion Formula

The final verdict combines three independent scoring mechanisms:

```
S_final = 0.50 × S_Neural + 0.20 × S_Heuristic + 0.30 × S_Manifold
```

**Multi-Gate Voting:**
- **3/3 agree** → "FAKE"
- **2/3 agree** → "SUSPICIOUS"
- **<2 agree** → "REAL"

---

## 🛠 Technology Stack

### Deep Learning & Audio
- **PyTorch 2.1.0** - Neural network framework
- **TorchAudio 2.1.0** - Audio processing & STFT
- **Transformers 4.57.0** - Pre-trained models (Wav2Vec2, HuBERT, Whisper, DeBERTa)
- **torchaudio** - Spectrogram, STFT, filtering

### ML & Signal Processing
- **scikit-learn 1.7.2** - Gaussian Mixture Models (GMM)
- **NumPy 2.2.6** - Numerical computing
- **SciPy 1.16.2** - Signal processing utilities

### Web Framework
- **FastAPI 0.128.0** - High-performance async web framework
- **Uvicorn 0.40.0** - ASGI server
- **WebSockets** - Real-time bidirectional communication

### Frontend
- **HTML5 + CSS3 + JavaScript** - Interactive dashboard
- **Canvas API** - Real-time waveform visualization

### Utilities
- **sounddevice 0.5.3** - Audio device interface

---

## ✨ Features

### Core Capabilities

🎙️ **Real-Time Voice Fraud Detection**
- Process audio streams at 48kHz with minimal latency
- WebSocket-based live feedback loop
- Processes 3-second audio chunks continuously

🧠 **Advanced Multi-Layer Analysis**
- Neural acoustic detection (WaveSpNetCore)
- Heuristic signal stability metrics
- Manifold anomaly detection via GMM
- Linguistic intent classification

🔊 **Linguistic Intent Recognition**
- Automatic Speech Recognition (Whisper ASR)
- Zero-shot intent classification (DeBERTa)
- Detects scam patterns: urgency, authority impersonation, financial demands

📊 **Comprehensive Telemetry**
- Fusion score with confidence breakdown
- Per-layer scores and voting agreement count
- Transcript with intent probabilities
- Full diagnostic data for forensics

🎨 **Interactive Web Dashboard**
- Live verdict display (REAL / SUSPICIOUS / FAKE)
- Waveform visualization
- Real-time probability tracking
- Intent breakdown charts
- Mic integration via MediaRecorder

---

## 📁 Project Structure

```
Vortex2.0-Hackathon/
├── app.py                  # FastAPI web server + WebSocket handler
├── pipeline.py             # VortexPipeline orchestration class
├── models.py               # All neural network architectures
├── train.py                # Training script with GMM fitting
├── ui.html                 # Interactive web dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── wavespnet_core.pth      # (Generated) Trained model weights
└── gmm_manifold.pkl        # (Generated) Trained GMM for manifold detection
```

### File Descriptions

| File | Purpose |
|------|---------|
| **app.py** | FastAPI server with WebSocket endpoint for streaming audio |
| **pipeline.py** | Main VortexPipeline class implementing the full detection pipeline |
| **models.py** | PyTorch neural network definitions (LWT, NCRA, Conformer, GRU, KAN, HybridVortexModel) |
| **train.py** | Training loop for WaveSpNetCore and GMM manifold fitting |
| **ui.html** | Single-page web app with real-time visualization and controls |
| **requirements.txt** | All Python package dependencies |

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional but recommended for GPU acceleration)
- 8GB+ RAM (16GB+ if using CUDA)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Naren-Kandasamy/Vortex2.0-Hackathon.git
cd Vortex2.0-Hackathon
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv vortex_env
source vortex_env/bin/activate  # On Windows: vortex_env\Scripts\activate

# Or using conda
conda create -n vortex python=3.10
conda activate vortex
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models (Optional but Recommended)
The first run will automatically download:
- Wav2Vec2 (facebook/wav2vec2-base-960h) ~360 MB
- HuBERT (facebook/hubert-base-ls960) ~360 MB
- Whisper (openai/whisper-tiny.en) ~139 MB
- DeBERTa (cross-encoder/nli-deberta-v3-small) ~270 MB

**Total: ~1.1 GB download**

```bash
# Pre-download to avoid delays on first run (optional)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/wav2vec2-base-960h')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/hubert-base-ls960')"
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-tiny.en')"
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-small')"
```

---

## 🚀 Usage

### Option 1: Run Web Application (Recommended)

```bash
python app.py
```

Output:
```
Loading VORTEX Pipeline models...
Loaded wavespnet_core.pth successfully.
Loaded wavespnet_gmm.pkl successfully.
Loading Whisper ASR (tiny.en)...
Loading DeBERTa Zero-Shot Intent Classifier...
VORTEX Engine Ready.

Uvicorn running on http://0.0.0.0:8000
```

Then open your browser to: **http://localhost:8000**

**UI Controls:**
- Click **"Start Recording"** to begin capturing audio
- System processes 3-second chunks in real-time
- Results display instantly with verdict and confidence scores
- Click **"Stop Recording"** to end session

### Option 2: Train Your Own Model

**Prerequisites:** Audio dataset in the following structure:
```
combined_folder/
├── real/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── fake/
    ├── fake1.wav
    ├── fake2.wav
    └── ...
```

**Run Training:**
```bash
python train.py
```

**Output:**
- `vortex_wavespnet_core.pt` - Trained model weights
- `gmm_manifold.pkl` - Fitted Gaussian Mixture Model

**Training Configuration:**
```python
train_model(
    epochs=5,           # Number of training epochs
    batch_size=8,       # Batch size
    learning_rate=1e-4  # Adam learning rate
)
```

---

## 🧠 How It Works

### Real-Time Processing Flow

```
User Speaks → Browser Records → WebSocket Sends (16-bit PCM)
                                        ↓
                            [Vortex Pipeline Processes]
                                        ↓
                    ┌─────────┬���───────────┬──────────┐
                    ↓         ↓            ↓          ↓
            [Neural Score] [Heuristic] [Manifold] [ASR/Intent]
                    ↓         ↓            ↓          ↓
                    └─────────┴────────────┴──────────┘
                                ↓
                    [Fusion Score Calculation]
                                ↓
                    [Multi-Gate Voting]
                                ↓
            JSON Telemetry → WebSocket → Browser → Dashboard Update
```

### Stage-by-Stage Breakdown

#### 1. Audio Preprocessing
- Convert 16-bit PCM bytes to float32 [-1, 1]
- Peak normalize to unit amplitude
- Pad or truncate to 48,000 samples (3 seconds @ 16kHz)

#### 2. Neural Analysis (S_Neural)
- Forward through FrozenFoundations → WaveSpNetCore
- Output: Probability [0, 1] that audio is fake

#### 3. Heuristic Analysis (S_Heuristic)
- **Micro-Timer Jitter**: Peak-to-peak energy stability
  - AI often too smooth or artifacting-prone
  - Score: 0.0 (normal) or 1.0 (anomalous)
- **Spectral Entropy**: Frequency distribution irregularity
  - Human speech = spiky entropy
  - AI synthesis = flat entropy
  - Score: 0.0-1.0 continuous range

#### 4. Manifold Analysis (S_Manifold)
- Extract HuBERT embeddings (768-dim)
- Evaluate log-probability from trained GMM
- Anomaly score via sigmoid: `1 / (1 + exp(log_prob + 80))`
- Score: 0.0 (normal) to 1.0 (anomalous)

#### 5. Fusion Score
```
S_final = 0.50 × S_Neural + 0.20 × S_Heuristic + 0.30 × S_Manifold
```

#### 6. Multi-Gate Verdict
- Count how many layers predict fake (> 0.5 threshold)
- **3/3 layers** → "FAKE" (high confidence)
- **2/3 layers** → "SUSPICIOUS" (mixed signals)
- **<2 layers** → "REAL" (passes most layers)

#### 7. Linguistic Analysis (Intent)
- **ASR**: Whisper transcribes audio
- **Intent Classification**: DeBERTa scores against scam labels
  - Urgency ("now", "immediately", "hurry")
  - Authority Impersonation ("FBI", "police", "official")
  - Financial Demand ("payment", "wire transfer", "gift card")

---

## 🤖 Model Components

### LearnableWaveletTransform (LWT)
- Learnable low/high-pass filters
- 8-subband decomposition
- Temporal CNN: 8 → 64 → 128 → 256 channels
- Output: [B, 256, 149]

### NeuralCodecResidualAnalyzer (NCRA)
- STFT magnitude spectrograms (513 freq bins)
- 4-layer CNN: 513 → 256 → 256 → 128 → 128
- GRU backend (input=128, hidden=256)
- Output: [B, 149, 256]

### ConformerBlock
- Self-attention (8-head)
- Depthwise separable convolution (kernel=31)
- Feed-forward networks (d_model → 2048 → d_model)
- Residual connections & layer normalization

### GRUBackend
- 2-layer GRU (input=256, hidden=256)
- Captures temporal dependencies
- Outputs last hidden state [B, 256]

### ExactKANLayer
- Kolmogorov-Arnold Network with spline basis functions
- Learnable basis centers and scales
- Non-linear function approximation

### HybridVortexModel (Full Stack)
- Composition of all above layers
- Frozen: Wav2Vec2, HuBERT
- Trainable: LWT, NCRA, Conformer, GRU, KAN
- ~2M trainable parameters (efficient!)

---

## 🎓 Training

### Dataset Requirements
- **Audio format**: WAV, MP3, FLAC
- **Sample rate**: 16 kHz (auto-resampled)
- **Duration**: 3+ seconds (randomly cropped to 3 sec)
- **Split**: 50/50 real vs fake audio
- **Recommended**: 100+ real samples, 100+ fake samples

### Training Pipeline

**Phase 1: Neural Core Training**
```
Epochs: 1-5
Batch Size: 8
Learning Rate: 1e-4
Optimizer: AdamW
Loss: Clamped Binary Cross-Entropy [1e-7, 1-1e-7]
```

- Freezes Wav2Vec2 and HuBERT
- Trains only WaveSpNetCore (~2M parameters)
- Uses data augmentation (random crops)
- Saves: `vortex_wavespnet_core.pt`

**Phase 2: Unsupervised GMM Fitting**
```
Real Embeddings: Extract HuBERT from REAL samples only
GMM Components: min(64, len(real_samples) / 2)
Covariance Type: Diagonal
```

- Fits 64-component Gaussian Mixture Model
- Models the "manifold" of legitimate human speech
- Detects embeddings far from this manifold
- Saves: `gmm_manifold.pkl`

### Custom Training

```python
from train import train_model

# Run with custom hyperparameters
train_model(
    epochs=10,
    batch_size=16,
    learning_rate=5e-5
)
```

---

## ⚙ Configuration

### Inference Parameters (in `pipeline.py`)

```python
# Audio preprocessing
TARGET_SIZE = 48000          # 3 seconds @ 16kHz
SAMPLE_RATE = 16000          # Input sample rate
PEAK_NORMALIZE = True        # Normalize to unit amplitude

# Fusion weights
NEURAL_WEIGHT = 0.50         # Neural score contribution
HEURISTIC_WEIGHT = 0.20      # Heuristic score contribution
MANIFOLD_WEIGHT = 0.30       # Manifold score contribution

# Voting thresholds
LAYER_THRESHOLD = 0.5        # Probability threshold per layer
FAKE_AGREEMENT = 3           # # layers for FAKE verdict
SUSPICIOUS_AGREEMENT = 2     # # layers for SUSPICIOUS verdict

# Heuristic parameters
JITTER_LO_BOUND = 0.0005     # Lower jitter bound
JITTER_HI_BOUND = 1.0        # Upper jitter bound
ENTROPY_THRESHOLD = 3.5      # Spectral entropy baseline
```

### WebSocket Configuration (in `app.py`)

```python
# CORS settings
ALLOWED_ORIGINS = ["*"]      # Adjust for production
ALLOW_CREDENTIALS = True
ALLOW_METHODS = ["*"]
ALLOW_HEADERS = ["*"]

# Server
HOST = "0.0.0.0"
PORT = 8000
```

---

## 🔌 WebSocket API

### Connection
```
ws://localhost:8000/stream
```

### Message Flow

**Client → Server (Audio Chunk)**
```
Binary frame: 16-bit PCM audio bytes (e.g., 48,000 samples = 96 KB per chunk)
Sample rate: 16 kHz
Format: Mono, little-endian
```

**Server → Client (Telemetry)**
```json
{
  "verdict": "REAL",  // "REAL", "SUSPICIOUS", or "FAKE"
  "fusion_score": 0.3245,
  "layers": {
    "neural": 0.2100,
    "heuristic": 0.0500,
    "manifold": 0.5400,
    "agreements": 1
  },
  "transcript": "Please verify your account",
  "intent_confidence": 0.8521,
  "intent_breakdown": {
    "urgency": 0.6200,
    "authority impersonation": 0.9100,
    "financial demand": 0.3400
  }
}
```

### Example Client (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:8000/stream");

ws.onopen = () => {
  console.log("Connected to Vortex");
  // Send audio chunks
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        ws.send(e.data);  // Send audio chunk
      };
      mediaRecorder.start(3000); // 3-sec chunks
    });
};

ws.onmessage = (event) => {
  const telemetry = JSON.parse(event.data);
  console.log("Verdict:", telemetry.verdict);
  console.log("Confidence:", telemetry.fusion_score);
  // Update UI...
};
```

---

## 📊 Performance Metrics

### Expected Performance
(Metrics from Hackathon MVP - trained on limited data)

| Metric | Value |
|--------|-------|
| **Latency** | ~200-500 ms per 3-sec chunk |
| **GPU Memory** | ~2.5 GB (with all models loaded) |
| **CPU Memory** | ~1 GB (inference) |
| **Throughput** | 6-15 chunks/sec (hardware dependent) |

### Model Sizes
| Model | Size |
|-------|------|
| Wav2Vec2 (frozen) | 360 MB |
| HuBERT (frozen) | 360 MB |
| WaveSpNetCore | 8.5 MB |
| GMM Manifold | 2.3 MB |
| Whisper ASR | 139 MB |
| DeBERTa Intent | 270 MB |
| **Total** | ~1.1 GB |

### Inference Breakdown (per 3-sec chunk)
- Audio preprocessing: ~5 ms
- Neural forward pass: ~80-150 ms
- Heuristic calculation: ~30-50 ms
- Manifold scoring: ~20-30 ms
- ASR (Whisper): ~200-400 ms
- Intent classification: ~50-100 ms
- **Total: 200-700 ms (depending on HW)**

---

## 🔧 Troubleshooting

### Issue: Models not loading on first run

**Solution**: Manually pre-download models:
```bash
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-tiny.en')"
```

### Issue: WebSocket connection fails

**Solution**: Check firewall and CORS settings:
```python
# In app.py, adjust CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specific origin
    ...
)
```

### Issue: Out of memory (OOM)

**Solution**:
1. Use CPU instead:
   ```bash
   CUDA_VISIBLE_DEVICES="" python app.py
   ```
2. Reduce batch size in `train.py`
3. Use smaller models (e.g., whisper-base instead of tiny)

### Issue: No audio detected

**Solution**:
1. Check browser microphone permissions
2. Verify audio device in browser's audio input settings
3. Test with pre-recorded audio file via WebSocket

### Issue: GMM not loading (S_Manifold always 0.0)

**Solution**: Train GMM first:
```bash
python train.py
```
Ensure you have real speech samples in `combined_folder/real/`

### Issue: Slow inference on CPU

**Expected**: CPU inference is ~5-10x slower than GPU.
**Solution**: Use NVIDIA GPU with CUDA:
```bash
# Automatically uses CUDA if available
python app.py
```

---

## 🔮 Future Enhancements

### Phase 2 (Post-Hackathon)
- [ ] Fine-tune Wav2Vec2 on fake detection task
- [ ] Multi-language support (DeBERTa multi-lingual)
- [ ] Streaming ASR (live transcription without buffering)
- [ ] Custom scam taxonomy training
- [ ] Speaker diarization for multi-speaker audio
- [ ] Explainability layer (SHAP/attention visualizations)

### Phase 3 (Production)
- [ ] TorchScript / ONNX export for edge deployment
- [ ] Mobile app integration (iOS/Android)
- [ ] Federated learning for privacy-preserving updates
- [ ] Real-time model serving (Triton, KServe)
- [ ] Database logging and analytics dashboard
- [ ] Integration with telephony platforms (Twilio, Asterisk)
- [ ] Block-list management and threat intelligence

### Phase 4 (Research)
- [ ] Open-source dataset contribution
- [ ] Peer-reviewed publication
- [ ] Benchmark against state-of-the-art (WaveFake, DXCPA, etc.)
- [ ] Robustness against adversarial attacks
- [ ] Zero-shot cross-domain generalization

---

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Improved model architectures
- Better training datasets
- Performance optimizations
- Bug fixes and documentation
- Alternative frontend implementations
- Deployment templates (Docker, K8s)

**Contribution Steps:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is part of the **Vortex 2.0 Hackathon**. Please refer to `LICENSE` file for details.

---

## 👨‍💻 Authors

**Naren Kandasamy**
- GitHub: [@Naren-Kandasamy](https://github.com/Naren-Kandasamy)
- Vortex 2.0 AI Voice Fraud Detection System

---

## 📞 Contact & Support

- **Issues**: GitHub Issues (bug reports, feature requests)
- **Discussions**: GitHub Discussions (Q&A, ideas)
- **Email**: For security concerns, please contact responsibly

---

## 🙏 Acknowledgments

- **Pre-trained Models**: Meta AI (Wav2Vec2, HuBERT), OpenAI (Whisper), Microsoft (DeBERTa)
- **Framework**: PyTorch, FastAPI, and open-source community
- **Hackathon**: Vortex 2.0 organizers and mentors
- **Inspiration**: Fighting voice fraud and protecting users from deepfake attacks

---

## 📚 References

### Academic Papers
- Conformer: Convolution-augmented Transformer for Speech Recognition
- Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- HuBERT: Self-supervised Speech Representation Learning by Masked Prediction of Hidden Units
- Kolmogorov-Arnold Networks

### Tools & Libraries
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Hub](https://huggingface.co/models)

---

**Last Updated**: March 2026
**Status**: Hackathon MVP (Active Development)
**Version**: 2.0
