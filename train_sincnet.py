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

datadir = Path("data")

# Remove or conditionally apply torch.compile()
# if torch.__version__ >= "2.0" and sys.version_info < (3, 12):
#     model = torch.compile(model)
# else:
#     print("torch.compile() is not available. Using the model without compilation.")

#-------------------------------------------
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
            start = torch.randint(t_min, t_max - chunk_len + 1, (1,)).item()
            chunk = signal[start:start + chunk_len]
        else:
            chunk = signal[t_min:t_max]
            # Pad if necessary
            if len(chunk) < chunk_len:
                # print(f"Padding chunk {file_path} which was {len(chunk) * 1000 / sample_rate:.2f} ms long")
                chunk = F.pad(chunk, (0, chunk_len - len(chunk)))

        # Apply random amplitude scaling
        # amp_scale = torch.FloatTensor(1).uniform_(1 - augment_factor, 1 + augment_factor).to(device)
        # chunk = chunk * amp_scale

        x.append(chunk)
        y.append(int(row['label']))

    return torch.stack(x).unsqueeze(1), torch.tensor(y, dtype=torch.long)

with open("mod_all_classes_test_files.csv", 'r') as csvfile:
    test_data_list = list(csv.DictReader(csvfile))

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
            f.write(f"{epoch * batches_per_epoch + batch_idx} train {loss.item():.4f}\n")
    
    avg_train_loss = train_loss / batches_per_epoch

    if not epoch % 8 == 0:
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

                chunk_len = (cfg.cw_len * cfg.sample_rate) // 1000 # chunk length in samples
                chunk_shift = (cfg.cw_shift * cfg.sample_rate) // 1000 # chunk shift in samples

                # calculate number of chunks
                num_chunks = int((signal.shape[0] - chunk_len) / chunk_shift) + 1
                chunks = signal.unfold(0, chunk_len, chunk_shift)
                pout = torch.zeros(num_chunks, cfg.num_classes).to(signal.device)

                for i in range(0, num_chunks, cfg.batch_size):
                    batch = chunks[i:min(i+cfg.batch_size, num_chunks)]
                    # Add an extra dimension to match the expected input shape
                    batch = batch.unsqueeze(1)  # Shape becomes [batch_size, 1, chunk_len]
                    pout[i:i+batch.shape[0]] = model(batch)

                # Calculate predictions and errors for the entire file
                pred = torch.argmax(pout, dim=1)
                lab = torch.full((num_chunks,), label).to(signal.device)
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

        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_end_time = time.time()
        eval_duration = epoch_end_time - eval_start_time
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {total_loss:.4f} | "
            f"Frame Accuracy: {1 - total_frame_error:.4f} | "
            f"Sentence Accuracy: {1 - total_sent_error:.4f} | "
            f"Eval Time: {eval_duration:.2f} seconds | "
            f"Epoch Time: {epoch_duration:.2f} seconds | ")
        
        # log epoch metrics
        with open(log_file, "a") as f:
            f.write(f"{(epoch + 1) * batches_per_epoch} val {total_loss:.4f}\n")
