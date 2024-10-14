import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf


class AudioDataset(Dataset):
    def __init__(self, root_dir: str, sample_rate: int = 44100, cw_len: int = 10, augment_factor: float = 0.1):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.chunk_length = int(cw_len * sample_rate / 1000)  # Convert ms to samples
        self.augment_factor = augment_factor
        self.files, self.labels = self._load_dataset()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def _load_dataset(self) -> Tuple[List[Path], List[str]]:
        files, labels = [], []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for file in class_dir.glob("*.wav"):
                    files.append(file)
                    labels.append(class_dir.name)
        return files, labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.files[idx]
        label = self.label_to_idx[self.labels[idx]]

        # Load audio using librosa
        # waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        waveform, sr = sf.read(audio_path, dtype='float32')
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, sr, self.sample_rate)
        if len(waveform.shape) > 1:
            waveform = librosa.to_mono(waveform.T)

        waveform = torch.from_numpy(waveform).unsqueeze(0)

        # Randomly select a chunk if the audio is longer than chunk_length
        if waveform.shape[1] > self.chunk_length:
            start = torch.randint(0, waveform.shape[1] - self.chunk_length, (1,))
            waveform = waveform[:, start:start+self.chunk_length]
        else:
            waveform = F.pad(waveform, (0, self.chunk_length - waveform.shape[1]))

        # Apply random amplitude scaling
        amp_scale = torch.FloatTensor(1).uniform_(1 - self.augment_factor, 1 + self.augment_factor)
        waveform = waveform * amp_scale

        # Normalize the waveform
        waveform = waveform / torch.max(torch.abs(waveform))

        return waveform, label

def create_dataloaders(root_dir: str, batch_size: int = 32, 
                       train_ratio: float = 0.8, sample_rate: int = 44100, 
                       cw_len: int = 10, augment_factor: float = 0.1):
    dataset = AudioDataset(root_dir, sample_rate, cw_len, augment_factor)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Use RandomSampler for training to ensure random batch creation
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
