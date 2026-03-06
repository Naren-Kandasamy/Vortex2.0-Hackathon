import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
import matplotlib.pyplot as plt
import seaborn as sns
from models import HybridVortexModel

class AudioDataset(Dataset):
    """
    Dataset using random 10-second crops if long, or padded to 10-sec
    (We will stick to the 3-second 48000 sample window for consistency across the architecture).
    """
    def __init__(self, data_dirs, target_samples=48000, sample_rate=16000):
        self.target_samples = target_samples
        self.sample_rate = sample_rate
        self.files = []
        
        for label_str, dirs in data_dirs.items():
            label = 0.0 if label_str == "real" else 1.0 # <=0.5 REAL, >0.5 FAKE
            for d in dirs:
                if not os.path.exists(d):
                    continue
                for f in os.listdir(d):
                    if f.endswith(('.wav', '.mp3', '.flac')):
                        self.files.append((os.path.join(d, f), label))
                        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        waveform, sr = torchaudio.load(file_path)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.squeeze(0)
        
        # Audio Augmentation: Random 3-second crop if file is longer
        if waveform.shape[0] > self.target_samples:
            max_start = waveform.shape[0] - self.target_samples
            start_idx = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[start_idx : start_idx + self.target_samples]
        elif waveform.shape[0] < self.target_samples:
            padding = self.target_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Peak Normalize
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
            
        waveform = waveform.to(torch.float32)
        label = torch.tensor([label], dtype=torch.float32)
        return waveform, label

def train_model(epochs=5, batch_size=8, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Balancing: 50/50 split of Real vs Fake
    data_dirs = {
        "real": ["combined_folder/real"],
        "fake": ["combined_folder/fake"]
    }
    
    dataset = AudioDataset(data_dirs)
    
    if len(dataset) == 0:
        print("No valid dataset found. Creating dummy dataset for validation of script...")
        class DummyDataset(Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                label = 1.0 if idx % 2 == 0 else 0.0
                return torch.randn(48000, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)
        dataset = DummyDataset()
        
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize Hybrid Stack
    model = HybridVortexModel(device=device).to(device)
    # Only optimize the WaveSpNetCore, foundations are frozen
    optimizer = torch.optim.AdamW(model.core.parameters(), lr=learning_rate)
    
    # Custom Loss: BCELoss clamped at 1e-7 for numeric stability (especially inside the KAN)
    def clamped_bce(pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        return nn.functional.binary_cross_entropy(pred, target)
        
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Forward pass: fake_prob and hubert_pooled. We only need fake_prob for supervised loss
            outputs, _ = model(waveforms) 
            loss = clamped_bce(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
    print("Neural core training complete. Saving weights...")
    torch.save(model.core.state_dict(), 'vortex_wavespnet_core.pt')
    
    # ---------------------------------------------------------------------------
    # Phase 2: Unsupervised GMM Fitting (The "Mathematical Fence")
    # ---------------------------------------------------------------------------
    print("Extracting HuBERT Manifold for GMM Fitting on REAL data only...")
    model.eval()
    real_embeddings = []
    
    with torch.no_grad():
        for waveforms, labels in train_loader:
            for i in range(len(labels)):
                if labels[i].item() == 0.0: # REAL
                    # Get the single waveform
                    wf = waveforms[i].unsqueeze(0).to(device)
                    # We bypass computing the probability, just get embedding
                    _, hubert_emb = model(wf) 
                    real_embeddings.append(hubert_emb.cpu().numpy().squeeze())
                    
    if len(real_embeddings) > 10 and HAS_SKLEARN:
        print(f"Fitting 64-Component GMM on {len(real_embeddings)} real audio samples...")
        embs = np.array(real_embeddings)
        # We might use fewer components if data is extremely small
        n_comp = min(64, len(real_embeddings) // 2)
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag')
        gmm.fit(embs)
        
        with open('gmm_manifold.pkl', 'wb') as f:
            pickle.dump(gmm, f)
        print("GMM Manifold Saved.")
    else:
        print("Skipped GMM Fitting (Not enough REAL data or sklearn missing)")

if __name__ == "__main__":
    train_model(epochs=1) # Run 1 epoch for Hackathon speed
