import os
import time
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import List, Tuple

import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from SincNetModel import SincNetModel

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

cfg = SincNetConfig()

model = SincNetModel(cfg).to(device)
print(f"Total number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
model = torch.compile(model)

datadir = Path(cfg.datadir)

# --- SETUP DATA -----------------------------------------------------------------------------------------------
cache = {}
def load_audio(file_path, device, sample_rate=44100):
    if file_path in cache:
        return cache[file_path]

    signal, fs = sf.read(file_path)

    if fs != sample_rate:
        raise ValueError(f"File {file_path} has sample rate {fs}, expected {sample_rate}")
    
    signal = torch.from_numpy(signal).float().to(device)

    # Ensure the signal is a 1D tensor
    if signal.dim() != 1:
        raise ValueError(f"Signal must be a 1D tensor. Found {signal.dim()}D tensor.")
    
    # Normalize the signal to be between -1 and 1
    signal = signal / torch.abs(signal.max())

    # Store the processed signal in the cache
    cache[file_path] = signal

    return signal

def create_train_batch(batch_size=128, datadir="data", data_list="mod_all_classes_train_files.csv", cw_len=10, sample_rate = 44100, augment_factor = 0.2):
    datadir = Path(datadir)
    chunk_len = (cw_len * sample_rate) // 1000 # chunk length in samples
    
    with open(data_list, 'r') as csvfile:
        data_list = list(csv.DictReader(csvfile))
    
    # select batch_size random files from the data_list
    indices = torch.randperm(len(data_list))[:batch_size]

    x = []
    y = []

    for idx in indices:
        row = data_list[idx]
        file_path = datadir / row['file']
        signal = load_audio(file_path, device, sample_rate)
        
        # calculate start and end of vocalization annotation
        t_min, t_max = int(float(row['start']) * sample_rate), int(float(row['start']) * sample_rate + float(row['length']) * sample_rate)
        
        # if the vocalization is longer than the chunk length, randomly select a start point
        if t_max - t_min > chunk_len:
            rand_start = torch.randint(t_min, t_max - chunk_len, (1,)).item()
            chunk = signal[rand_start:rand_start+chunk_len]
        else:
            # TODO: fix this to sample from the middle of the vocalization instead of padding
            # but not that important, since it is around 50 examples per epoch
            chunk = signal[t_min:t_max]
            # Pad if necessary
            if len(chunk) < chunk_len:
                # print(f"Padding chunk {file_path} which was {len(chunk) * 1000 / sample_rate:.2f} ms long")
                chunk = F.pad(chunk, (0, chunk_len - len(chunk)))

        # Apply random amplitude scaling
        amp_scale = torch.FloatTensor(1).uniform_(1 - augment_factor, 1 + augment_factor).to(device)
        chunk = chunk * amp_scale

        x.append(chunk)
        y.append(int(row['label']))

    return torch.stack(x).unsqueeze(1), torch.tensor(y, dtype=torch.long)

with open("mod_all_classes_test_files.csv", 'r') as csvfile:
    test_data_list = list(csv.DictReader(csvfile))

# --- TRAINING --------------------------------------------------------------------------------------------------
log_file = os.path.join(os.path.dirname(__file__), "trainlog.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

for epoch in range(cfg.num_epochs):
    model.train()
    train_loss = 0
    
    epoch_start_time = time.time()

    # Training loop
    for batch_idx in range(cfg.batches_per_epoch):
        x, y = create_train_batch(batch_size=cfg.batch_size, sample_rate=cfg.sample_rate, cw_len=cfg.cw_len)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
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

        total_loss = 0
        total_frame_error = 0
        total_sent_error = 0

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

                total_loss += loss.item()
                total_frame_error += frame_error.item()
                total_sent_error += sentence_error.item()

        total_loss /= len(test_data_list)
        total_frame_error /= len(test_data_list)
        total_sent_error /= len(test_data_list)

        #scheduler.step(total_loss)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_end_time = time.time()
        eval_duration = epoch_end_time - eval_start_time
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {total_loss:.4f} | "
            f"Sentence Accuracy: {1 - total_sent_error:.4f} | "
            f"Frame Accuracy: {1 - total_frame_error:.4f} | "
            #f"Eval Time: {eval_duration:.2f} seconds | "
            f"Epoch Time: {epoch_duration:.2f} seconds | ")
            #f"LR: {scheduler.get_last_lr()[0]:.8f}")
        
        # log epoch metrics
        with open(log_file, "a") as f:
            f.write(f"{(epoch + 1) * cfg.batches_per_epoch} val {total_loss:.4f}\n")
