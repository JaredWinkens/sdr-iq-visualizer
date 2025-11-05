import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sigmf
from sigmf import SigMFFile, sigmffile
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# ============================================
# SigMF Dataset Loader for PyTorch
# ============================================

class SigMFDataset(Dataset):
    """
    PyTorch Dataset for SigMF files
    Handles loading, windowing, and preprocessing of SigMF data
    """
    def __init__(
        self, 
        sigmf_files: List[str],
        window_size: int = 128,
        stride: int = 64,
        normalize: bool = True,
        augment: bool = False,
        target_snr_db: float = None
    ):
        """
        Args:
            sigmf_files: List of paths to .sigmf-meta files
            window_size: Number of IQ samples per window
            stride: Stride for sliding window (overlap if < window_size)
            normalize: Whether to normalize IQ samples
            augment: Whether to apply data augmentation
            target_snr_db: If set, adds noise to achieve target SNR
        """
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.augment = augment
        self.target_snr_db = target_snr_db
        
        self.samples = []
        self.labels = []
        self.metadata = []
        
        # Load all SigMF files
        for sigmf_path in sigmf_files:
            self._load_sigmf_file(sigmf_path)
        
        print(f"Loaded {len(self.samples)} windows from {len(sigmf_files)} SigMF files")
    
    def _load_sigmf_file(self, meta_path: str):
        """Load a single SigMF file and extract windows"""
        # Load SigMF metadata
        signal = sigmf.sigmffile.fromfile(meta_path)
        
        # Get IQ samples
        samples = signal.read_samples()
        
        # Get annotations (labels)
        annotations = signal.get_annotations()
        
        # Extract metadata
        global_meta = signal.get_global_info()
        sample_rate = global_meta.get('core:sample_rate', 1e6)
        frequency = global_meta.get('core:frequency', 0)
        
        # Create windows from the signal
        for annotation in annotations:
            start_idx = annotation.get('core:sample_start', 0)
            count = annotation.get('core:sample_count', len(samples))
            label = annotation.get('core:label', 'unknown')
            
            # Extract the annotated segment
            segment = samples[start_idx:start_idx + count]
            
            # Create sliding windows
            for i in range(0, len(segment) - self.window_size, self.stride):
                window = segment[i:i + self.window_size]
                
                if len(window) == self.window_size:
                    self.samples.append(window)
                    self.labels.append(label)
                    self.metadata.append({
                        'sample_rate': sample_rate,
                        'frequency': frequency,
                        'start_idx': start_idx + i,
                        'file': meta_path
                    })
    
    def _normalize_iq(self, iq_samples):
        """Normalize IQ samples to unit power"""
        power = np.mean(np.abs(iq_samples) ** 2)
        if power > 0:
            return iq_samples / np.sqrt(power)
        return iq_samples
    
    def _add_noise(self, iq_samples, snr_db):
        """Add AWGN to achieve target SNR"""
        signal_power = np.mean(np.abs(iq_samples) ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(iq_samples)) + 1j * np.random.randn(len(iq_samples))
        )
        return iq_samples + noise
    
    def _augment_sample(self, iq_samples):
        """Apply random augmentations"""
        # Random phase rotation
        if np.random.rand() > 0.5:
            phase = np.random.uniform(0, 2 * np.pi)
            iq_samples = iq_samples * np.exp(1j * phase)
        
        # Random frequency offset
        if np.random.rand() > 0.5:
            freq_offset = np.random.uniform(-0.1, 0.1)
            t = np.arange(len(iq_samples))
            iq_samples = iq_samples * np.exp(1j * 2 * np.pi * freq_offset * t)
        
        # Random time shift
        if np.random.rand() > 0.5:
            shift = np.random.randint(-10, 10)
            iq_samples = np.roll(iq_samples, shift)
        
        return iq_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get IQ samples
        iq_samples = self.samples[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation
        if self.augment:
            iq_samples = self._augment_sample(iq_samples)
        
        # Add noise if target SNR specified
        if self.target_snr_db is not None:
            iq_samples = self._add_noise(iq_samples, self.target_snr_db)
        
        # Normalize
        if self.normalize:
            iq_samples = self._normalize_iq(iq_samples)
        
        # Convert to real-valued representation [I, Q]
        # Shape: (2, window_size)
        iq_tensor = torch.stack([
            torch.tensor(iq_samples.real, dtype=torch.float32),
            torch.tensor(iq_samples.imag, dtype=torch.float32)
        ])
        
        return iq_tensor, label


# ============================================
# Label Encoder for String Labels
# ============================================

class LabelEncoder:
    """Encode string labels to integers and vice versa"""
    def __init__(self):
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_classes = 0
    
    def fit(self, labels: List[str]):
        """Build encoding from list of labels"""
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        print(f"Found {self.num_classes} classes: {unique_labels}")
        return self
    
    def transform(self, labels: List[str]) -> torch.Tensor:
        """Convert labels to indices"""
        return torch.tensor([self.label_to_idx[label] for label in labels])
    
    def inverse_transform(self, indices: torch.Tensor) -> List[str]:
        """Convert indices back to labels"""
        return [self.idx_to_label[idx.item()] for idx in indices]


# ============================================
# Custom Collate Function
# ============================================

def sigmf_collate_fn(batch, label_encoder):
    """Custom collate function to handle string labels"""
    iq_tensors = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    label_indices = label_encoder.transform(labels)
    return iq_tensors, label_indices


# ============================================
# IQEngine Dataset Downloader
# ============================================

class IQEngineDownloader:
    """
    Helper class to download datasets from IQEngine
    """
    @staticmethod
    def list_available_datasets():
        """List some popular datasets available on IQEngine"""
        datasets = {
            'wifi': 'WiFi signals at various modulations',
            'bluetooth': 'Bluetooth Low Energy signals',
            'lte': 'LTE cellular signals',
            'radar': 'Radar pulses and chirps',
            'satellite': 'Satellite downlink signals',
            'amateur_radio': 'Ham radio communications'
        }
        print("Available datasets on IQEngine:")
        for name, desc in datasets.items():
            print(f"  - {name}: {desc}")
        print("\nVisit https://iqengine.org/browser to browse and download")
    
    @staticmethod
    def download_from_iqengine(dataset_url: str, output_dir: str = './data'):
        """
        Download a dataset from IQEngine
        
        Args:
            dataset_url: URL to the dataset on IQEngine
            output_dir: Local directory to save files
        """
        # This would use the IQEngine API or direct downloads
        # For now, users should manually download from the website
        print(f"To download datasets:")
        print(f"1. Visit {dataset_url}")
        print(f"2. Click 'Download' for .sigmf-meta and .sigmf-data files")
        print(f"3. Save both files to {output_dir}/")
        print(f"4. Ensure filenames match (same basename)")


# ============================================
# Complete Training Example
# ============================================

def train_with_sigmf_data():
    """Complete example of training with SigMF data"""
    
    # Step 1: Specify your SigMF files
    # These should be paths to .sigmf-meta files
    data_dir = Path('./sigmf_data')
    sigmf_files = list(data_dir.glob('*.sigmf-meta'))
    
    print(f"Found {len(sigmf_files)} SigMF files in {data_dir}")
    
    if len(sigmf_files) == 0:
        print("\n⚠️  No SigMF files found!")
        print("Download datasets from https://iqengine.org/browser")
        IQEngineDownloader.list_available_datasets()
        return
    
    # Step 2: Create dataset
    dataset = SigMFDataset(
        sigmf_files=[str(f) for f in sigmf_files],
        window_size=128,
        stride=64,
        normalize=True,
        augment=True,
        target_snr_db=10  # Add noise for 10 dB SNR
    )
    
    # Step 3: Fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset.labels)
    
    # Step 4: Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Step 5: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=lambda batch: sigmf_collate_fn(batch, label_encoder),
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda batch: sigmf_collate_fn(batch, label_encoder),
        num_workers=4
    )
    
    # Step 6: Create model (using ResNet from previous response)
    from model import RadioSignalResNet, RadioSignalTrainer
    
    model = RadioSignalResNet(num_classes=label_encoder.num_classes)
    
    # Step 7: Train
    trainer = RadioSignalTrainer(model)
    trainer.train(train_loader, val_loader, epochs=50)
    
    print("\n✓ Training complete!")
    print(f"Model saved to: best_model.pth")
    print(f"Classes: {label_encoder.idx_to_label}")


# ============================================
# Visualization Helpers
# ============================================

def visualize_sigmf_sample(dataset, idx):
    """Visualize a sample from the dataset"""
    iq_tensor, label = dataset[idx]
    iq_samples = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain
    axes[0, 0].plot(iq_samples.real, label='I')
    axes[0, 0].plot(iq_samples.imag, label='Q')
    axes[0, 0].set_title(f'Time Domain - {label}')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].legend()
    
    # Constellation
    axes[0, 1].scatter(iq_samples.real, iq_samples.imag, alpha=0.5, s=1)
    axes[0, 1].set_title('Constellation Diagram')
    axes[0, 1].set_xlabel('In-phase')
    axes[0, 1].set_ylabel('Quadrature')
    axes[0, 1].grid(True)
    
    # FFT
    fft = np.fft.fftshift(np.fft.fft(iq_samples))
    axes[1, 0].plot(20 * np.log10(np.abs(fft) + 1e-10))
    axes[1, 0].set_title('Frequency Domain (FFT)')
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    
    # Spectrogram
    axes[1, 1].specgram(iq_samples, Fs=1, NFFT=32, noverlap=16)
    axes[1, 1].set_title('Spectrogram')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'sigmf_sample_{label}_{idx}.png', dpi=150)
    plt.show()


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    print("SigMF PyTorch Training Pipeline")
    print("=" * 60)
    
    # Option 1: List available datasets
    IQEngineDownloader.list_available_datasets()
    
    # Option 2: Train with your downloaded SigMF data
    train_with_sigmf_data()