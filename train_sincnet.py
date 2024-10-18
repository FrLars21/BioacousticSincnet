import os
import time
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import List, Tuple
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp

import torchaudio

from SincNetModel import SincNetModel

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

scaler = amp.GradScaler()

print(f"Using device: {device}")

# --- SETUP MODEL AND CONFIGURATION -------------------------------------------------------------------------------
@dataclass
class SincNetConfig:

    # Data
    datadir: str = "data"

    # Optimization
    batch_size: int = 256
    batches_per_epoch: int = 80
    num_epochs: int = 400

    # Windowing parameters
    sample_rate: int = 44100
    cw_len: int = 20 # window length in ms
    cw_shift: int = 1 # overlap in ms

    # number of filters, kernel size (filter length), stride
    conv_layers: List[Tuple[int, int, int]] = (
        (80, 125, 1),
        (60, 5, 1),
        (60, 5, 1),
    )
    # set any of these <= 1 to disable max pooling for the conv layer
    conv_max_pool_len: Tuple[int] = (3, 3, 3)

    # if batchnorm is not used, layernorm is applied instead
    conv_layers_batchnorm: bool = True

    # number of neurons
    fc_layers: List[int] = (
        768,
        768,
        1024,
    )

    fc_layers_batchnorm: bool = True

    # number of classes
    num_classes: int = 87

with open('config.yml', 'r') as f: cfg_dict = yaml.safe_load(f)
cfg = SincNetConfig(**{k: v for k, v in cfg_dict.items() if k in SincNetConfig.__annotations__})

# --- HF datasets -----------------------------------------------------------------------------------------------
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader

train_ds = load_dataset("DBD-research-group/BirdSet", "NBP", split="train")
test_ds = load_dataset("DBD-research-group/BirdSet", "NBP", split="test_5s")

train_ds = train_ds.cast_column("audio", Audio(sampling_rate=32000))
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=32000))

def map_first_five(sample):
    max_length = 160000 # 32_000hz*5sec
    sample["audio"]["array"] =  sample["audio"]["array"][:max_length]
    return sample

train_ds = train_ds.map(map_first_five, batch_size=1000, num_proc=4)
test_ds = test_ds.map(map_first_five, batch_size=1000, num_proc=4)

train_ds = train_ds.with_format("torch")
test_ds = test_ds.with_format("torch")

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

print(next(iter(train_loader)))
print(next(iter(test_loader)))

import sys; sys.exit()

# --- SETUP MODEL -----------------------------------------------------------------------------------------------

model = SincNetModel(cfg).to(device)
print(f"Total number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
model = torch.compile(model)

# --- TRAINING --------------------------------------------------------------------------------------------------
log_file = os.path.join(os.path.dirname(__file__), "trainlog.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(cfg.num_epochs):
    model.train()
    train_loss = 0
    
    epoch_start_time = time.time()

    # Training loop
    for batch_idx in range(cfg.batches_per_epoch):
        x, y = create_train_batch(batch_size=cfg.batch_size, sample_rate=cfg.sample_rate, cw_len=cfg.cw_len)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with amp.autocast(dtype=torch.bfloat16):
            outputs = model(x)
            loss = criterion(outputs, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # log train loss per step
        with open(log_file, "a") as f:
            f.write(f"{epoch * cfg.batches_per_epoch + batch_idx} train {loss.item():.4f}\n")
    
    avg_train_loss = train_loss / cfg.batches_per_epoch

    #if not epoch % 8 == 0:
    if not True:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_end_time = time.time()
        print("Epoch done in {:.2f} seconds, train loss: {:.4f}".format(epoch_end_time - epoch_start_time, avg_train_loss))
    else:
        eval_start_time = time.time()
        
        # Validation loop
        model.eval()

        val_loss = 0

        with torch.no_grad():
            for val_file in test_data_list:
                label = int(val_file['label'])
                signal = load_audio(datadir / val_file['file'], device, cfg.sample_rate)

                # only use the annotated part of the signal
                t_min, t_max = int(float(val_file['start']) * cfg.sample_rate), int(float(val_file['start']) * cfg.sample_rate + float(val_file['length']) * cfg.sample_rate)
                signal = signal[t_min:t_max]

                chunk_len = (cfg.cw_len * cfg.sample_rate) // 1000  # chunk length in samples

                # Handle case where signal is shorter than chunk_len
                if signal.shape[0] < chunk_len:
                    # Pad the signal to chunk_len
                    signal = F.pad(signal, (0, chunk_len - signal.shape[0]))
                    chunks = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                else:
                    # Create chunks, including any remainder
                    num_chunks = (signal.shape[0] + chunk_len - 1) // chunk_len  # Ceiling division
                    padded_signal = F.pad(signal, (0, num_chunks * chunk_len - signal.shape[0]))
                    chunks = padded_signal.view(num_chunks, 1, chunk_len)

                # Process in batches
                pout = torch.zeros(chunks.shape[0], cfg.num_classes).to(signal.device)
                for i in range(0, chunks.shape[0], cfg.batch_size):
                    batch = chunks[i:min(i+cfg.batch_size, chunks.shape[0])]
                    pout[i:i+batch.shape[0]] = model(batch)

                # Calculate predictions and errors for the entire file
                pred = torch.argmax(pout, dim=1)
                lab = torch.full((chunks.shape[0],), label).to(signal.device)
                loss = F.cross_entropy(pout, lab)
                frame_error = (pred != lab).float().mean()
                
                # Calculate sentence-level prediction
                sentence_pred = torch.argmax(pout.sum(dim=0))
                sentence_error = (sentence_pred != lab[0]).float()

                val_loss += loss.item()
                total_frame_error += frame_error.item()
                total_sent_error += sentence_error.item()

        val_loss /= len(test_data_list)
        total_frame_error /= len(test_data_list)
        total_sent_error /= len(test_data_list)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_end_time = time.time()
        eval_duration = epoch_end_time - eval_start_time
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {1 - total_sent_error:.4f} | "
            #f"Eval Time: {eval_duration:.2f} seconds | "
            f"Epoch Time: {epoch_duration:.2f} seconds | ")
            #f"LR: {scheduler.get_last_lr()[0]:.8f}")
        
        # log epoch metrics
        with open(log_file, "a") as f:
            f.write(f"{(epoch + 1) * cfg.batches_per_epoch} val {total_loss:.4f}\n")
