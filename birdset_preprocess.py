from datasets import load_dataset, Audio
import torchaudio
import torch
import numpy as np

train_ds = load_dataset("DBD-research-group/BirdSet", "HSN", split="train")
test_ds = load_dataset("DBD-research-group/BirdSet", "HSN", split="test_5s")

# the dataset comes without an automatic Audio casting, this has to be enabled via huggingface
# this means that each time a sample is called, it is decoded (which may take a while if done for the complete dataset)
# in BirdSet, this is all done on-the-fly during training and testing (since the dataset size would be too big if mapping and saving it only once)
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=None))  # Allow any sampling rate
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=None))

def trim_and_resample(sample):
    array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    
    # Trim to 5 seconds
    max_length = 5 * sr
    trimmed = array[:max_length]
    
    # Pad if necessary
    if len(trimmed) < max_length:
        trimmed = np.pad(trimmed, (0, max_length - len(trimmed)))
    
    # Resample to 32kHz
    resampler = torchaudio.transforms.Resample(sr, 32000)
    resampled = resampler(torch.tensor(trimmed).float()).numpy()
    
    sample["audio"] = {"array": resampled, "sampling_rate": 32000}
    return sample

# Apply trimming and resampling to both datasets
train_ds = train_ds.map(trim_and_resample, batch_size=1000, num_proc=4)
test_ds = test_ds.map(trim_and_resample, batch_size=1000, num_proc=4)

print(train_ds)
print(test_ds)
