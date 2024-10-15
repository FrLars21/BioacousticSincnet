import os
import time
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import soundfile as sf

from SincNetModel import SincNetModel, SincNetConfig
from torch.utils.data import DataLoader, TensorDataset

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up model and configuration
cfg = SincNetConfig()
model = SincNetModel(cfg).to(device)

# Remove or conditionally apply torch.compile()
# if torch.__version__ >= "2.0" and sys.version_info < (3, 12):
#     model = torch.compile(model)
# else:
#     print("torch.compile() is not available. Using the model without compilation.")

class DataLoaderLite:
    def __init__(self, batch_size, datadir, data_list, sample_rate, cw_len, augment_factor = 0.2):

        self.augment_factor = augment_factor

        self.batch_size = batch_size
        self.datadir = Path(datadir)
        self.sample_rate = sample_rate

        self.chunk_length = int(cw_len * sample_rate / 1000)  # Convert ms to samples

        with open(data_list, 'r') as csvfile:
            self.data_list = list(csv.DictReader(csvfile))
        self.num_samples = len(self.data_list)
        
        self.cache = {}  # Cache for loaded audio files

    def __len__(self):
        return self.num_samples

    def load_audio(self, file_path):
        if file_path not in self.cache:
            signal, fs = sf.read(str(file_path))
            
            # Ensure the signal is single-channel
            if signal.ndim == 2:
                print(f"WARNING: converting stereo to mono: {file_path}")
                signal = signal.mean(axis=1)  # Convert stereo to mono by averaging channels
            elif signal.ndim > 2:
                raise ValueError(f"Unexpected number of dimensions in audio file: {file_path}")

            # Convert to tensor
            signal = torch.tensor(signal, dtype=torch.float32)
            
            # Normalize signal
            signal = signal / torch.abs(signal.max())
            
            self.cache[file_path] = signal

        return self.cache[file_path]

    def get_chunk(self, signal, start, end):
        if end - start > self.chunk_length:
            start = torch.randint(start, end - self.chunk_length + 1, (1,)).item()
            chunk = signal[start:start + self.chunk_length]
        else:
            chunk = signal[start:end]
            # Pad if necessary
            if len(chunk) < self.chunk_length:
                chunk = F.pad(chunk, (0, self.chunk_length - len(chunk)))

        return chunk

    def next_batch(self):
        batch_inputs = []
        batch_labels = []

        indices = torch.randperm(self.num_samples)[:self.batch_size]

        for idx in indices:
            row = self.data_list[idx % self.num_samples]
            file_path = self.datadir / row['file']
            signal = self.load_audio(file_path)

            t_min, t_max = int(float(row['start']) * self.sample_rate), int(float(row['start']) * self.sample_rate + float(row['length']) * self.sample_rate)
            
            chunk = self.get_chunk(signal, t_min, t_max)

            # Apply random amplitude scaling
            amp_scale = torch.FloatTensor(1).uniform_(1 - self.augment_factor, 1 + self.augment_factor)
            chunk = chunk * amp_scale

            batch_inputs.append(chunk)
            batch_labels.append(int(row['label']))

        return torch.stack(batch_inputs).unsqueeze(1), torch.tensor(batch_labels, dtype=torch.long)

# Data loading
datadir = Path("data")
train_loader = DataLoaderLite(batch_size=cfg.batch_size, datadir=datadir, data_list="mod_all_classes_train_files.csv", 
                              sample_rate=cfg.sample_rate, cw_len=cfg.cw_len)

# Load the validation set
inputs, labels = torch.load('validation_set.pt')

# Create a TensorDataset from inputs and labels
val_dataset = TensorDataset(inputs, labels)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

# Print the number of parameters in the model
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters in the model: {trainable_params:,}")

log_file = os.path.join(os.path.dirname(__file__), "trainlog.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())

num_epochs = cfg.num_epochs
batches_per_epoch = cfg.batches_per_epoch

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    epoch_start_time = time.time()

    # Training loop
    for batch_idx in range(batches_per_epoch):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # log train loss per step
        with open(log_file, "a") as f:
            f.write(f"{epoch * batches_per_epoch + batch_idx} train {loss.item():.4f}\n")
    
    avg_train_loss = train_loss / batches_per_epoch
    
    # Validation loop
    model.eval()
    val_loss = 0
    val_frame_accuracy = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            
            frame_predictions = torch.argmax(outputs, dim=1)
            val_frame_accuracy += (frame_predictions == batch_labels).float().mean().item()
            num_val_batches += 1

        val_loss /= num_val_batches
        val_frame_accuracy /= num_val_batches

        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Frame Accuracy: {val_frame_accuracy:.4f} | "
              f"Epoch Time: {epoch_duration:.2f} seconds")
            
        # log epoch metrics
        with open(log_file, "a") as f:
            f.write(f"{(epoch + 1) * batches_per_epoch} val {val_loss:.4f}\n")
