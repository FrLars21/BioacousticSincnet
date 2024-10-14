import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import random
import math
from collections import Counter


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
                       train_ratio: float = 0.75, sample_rate: int = 44100, 
                       cw_len: int = 10, augment_factor: float = 0.1,
                       random_state: int = 1313):
    dataset = AudioDataset(root_dir, sample_rate, cw_len, augment_factor)
    
    if random_state is not None:
        random.seed(random_state)
    
    labels = [dataset.labels[i] for i in range(len(dataset))]
    label_counts = Counter(labels)
    
    train_indices, test_indices = [], []
    for label, count in label_counts.items():
        label_indices = [i for i, l in enumerate(labels) if l == label]
        random.shuffle(label_indices)
        
        n_train = max(1, math.floor(train_ratio * count))  # Ensure at least 1 sample in train
        n_test = count - n_train
        
        if n_test == 0:  # If only 1 sample, put it in both train and test
            train_indices.extend(label_indices)
            test_indices.extend(label_indices)
        else:
            train_indices.extend(label_indices[:n_train])
            test_indices.extend(label_indices[n_train:])
    
    # Shuffle the train and test indices
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Create Subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Use RandomSampler for training to ensure random batch creation
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
