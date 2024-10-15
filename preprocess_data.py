from pathlib import Path
import csv

import torch
import torch.nn.functional as F
import soundfile as sf

def preprocess_train_data(data_list_path, datadir, device, sample_rate = 44100):

    all_file_ids = []
    all_labels = []
    all_signals = []

    with open(data_list_path, 'r') as csvfile:
        data_list = list(csv.DictReader(csvfile))

    for rowid, row in enumerate(data_list):
        label = int(row["label"])        
        signal, fs = sf.read(datadir / row["file"])

        if fs != sample_rate:
            raise ValueError(f"File {row['file']} has sample rate {fs}, expected {sample_rate}")
        
        signal = torch.from_numpy(signal).float().to(device)

        # Ensure the signal is a 1D tensor
        if signal.dim() != 1:
            raise ValueError(f"Signal must be a 1D tensor. Found {signal.dim()}D tensor.")
        
        # Normalize the signal to be between -1 and 1
        signal = signal / torch.abs(signal.max())

        all_file_ids.append(rowid)
        all_labels.append(label)
        all_signals.append(signal)
    
    # concatenate all the tensors
    all_file_ids = torch.tensor(all_file_ids, dtype=torch.long, device=device)
    all_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

    torch.save((all_file_ids.cpu(), all_signals, all_labels.cpu()), 'train_set.pt')
    print(f"Train set saved as 'train_set.pt'")
    print(all_file_ids.shape)
    print(all_labels.shape)
    print(len(all_signals))

    return all_file_ids, all_labels, all_signals

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datadir = Path("data")
    data_list = "mod_all_classes_test_files.csv"

    preprocess_train_data(data_list, datadir, device)