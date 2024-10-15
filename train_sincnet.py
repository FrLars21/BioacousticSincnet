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
                chunk = F.pad(chunk, (0, chunk_len - len(chunk)))

        # Apply random amplitude scaling
        amp_scale = torch.FloatTensor(1).uniform_(1 - augment_factor, 1 + augment_factor).to(device)
        chunk = chunk * amp_scale

        x.append(chunk)
        y.append(int(row['label']))

    return torch.stack(x).unsqueeze(1), torch.tensor(y, dtype=torch.long)

# # Data loading
# train_loader = DataLoaderLite(batch_size=cfg.batch_size, datadir=datadir, data_list="mod_all_classes_train_files.csv", 
#                               sample_rate=cfg.sample_rate, cw_len=cfg.cw_len)

# Load the validation set
inputs, labels, file_ids = torch.load('validation_set.pt')
inputs, labels, file_ids = inputs.to(device), labels.to(device), file_ids.to(device)

# Create a TensorDataset from inputs, labels, and file_ids
val_dataset = TensorDataset(inputs, labels, file_ids)
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

    eval_start_time = time.time()
    
    # Validation loop
    model.eval()
    val_loss = 0
    val_frame_accuracy = 0

    file_predictions = {}
    file_labels = {}
    #file_losses = {}

    with torch.no_grad():
        for x, y, file_id in val_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_frame_accuracy /= len(val_loader)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    epoch_end_time = time.time()
    eval_duration = epoch_end_time - eval_start_time
    epoch_duration = epoch_end_time - epoch_start_time
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Frame Accuracy: {val_frame_accuracy:.4f} | "
          f"Eval Time: {eval_duration:.2f} seconds | "
          f"Epoch Time: {epoch_duration:.2f} seconds | ")
    
    # log epoch metrics
    with open(log_file, "a") as f:
        f.write(f"{(epoch + 1) * batches_per_epoch} val {val_loss:.4f}\n")
