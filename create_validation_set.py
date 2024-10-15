from pathlib import Path
import csv
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def create_validation_set(data_list_path, datadir, sample_rate, chunk_length, chunk_shift, device):
    """Pre-compute the validation set with corresponding labels."""

    all_chunks = []
    all_labels = []

    with open(data_list_path, 'r') as csvfile:
        data_list = list(csv.DictReader(csvfile))

    for row in data_list:
        label = int(row["label"])
        signal_path = datadir / row["file"]
        
        signal, fs = sf.read(signal_path)

        # # Add this check:
        if fs != sample_rate:
            raise ValueError(f"File {signal_path} has sample rate {fs}, expected {sample_rate}")
        
        signal = torch.from_numpy(signal).float().to(device)

        # Ensure the signal is a 1D tensor
        if signal.dim() != 1:
            raise ValueError(f"Signal must be a 1D tensor. Found {signal.dim()}D tensor.")
        
        # Normalize the signal to be between -1 and 1
        signal = signal / torch.abs(signal.max())

        signal_length = signal.size(0)

        # Calculate the number of chunks
        num_chunks = 1 + (signal_length - chunk_length) // chunk_shift
        if (signal_length - chunk_length) % chunk_shift != 0:
            num_chunks += 1  # Include the last incomplete chunk

        if num_chunks <= 0:
            # Signal is shorter than one chunk; pad it
            padded_signal = F.pad(signal, (0, chunk_length - signal_length), 'constant', 0)
            chunks = padded_signal.unsqueeze(0)  # Shape: (1, chunk_length)
        else:
            # Calculate the total required length after padding
            total_length = (num_chunks - 1) * chunk_shift + chunk_length
            if signal_length < total_length:
                padding_needed = total_length - signal_length
                signal = F.pad(signal, (0, padding_needed), 'constant', 0)

            # Use unfold to create the chunks
            chunks = signal.unfold(0, chunk_length, chunk_shift)  # Shape: (num_chunks, chunk_length)

        # Reshape chunks to (num_chunks, 1, chunk_length)
        chunks = chunks.unsqueeze(1)

        # Normalize each chunk
        chunks = (chunks - chunks.mean(dim=2, keepdim=True)) / (chunks.std(dim=2, keepdim=True) + 1e-8)

        all_chunks.append(chunks)
        all_labels.append(torch.full((chunks.size(0),), label, dtype=torch.long, device=device))

    # Concatenate all chunks and labels into single tensors
    if all_chunks and all_labels:
        all_chunks = torch.cat(all_chunks, dim=0)  # Shape: (total_chunks, 1, chunk_length)
        all_labels = torch.cat(all_labels, dim=0)  # Shape: (total_chunks,)

        # Save the chunks and labels as a .pt file
        torch.save((all_chunks.cpu(), all_labels.cpu()), 'validation_set.pt')
        
        print(f"Validation set saved as 'validation_set.pt'")

        print(all_chunks.shape)
        print(all_labels.shape)
    else:
        print("No chunks were created. Please check the input data.")

    return all_chunks, all_labels

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datadir = Path("data")
    data_list = "mod_all_classes_test_files.csv"
    sample_rate = 44100

    cw_len = 18
    cw_shift = 6

    chunk_length = int(sample_rate * cw_len / 1000)
    chunk_shift = int(sample_rate * cw_shift / 1000)

    create_validation_set(data_list, datadir, sample_rate, chunk_length, chunk_shift, device)
