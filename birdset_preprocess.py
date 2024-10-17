from datasets import load_dataset, Audio

train_ds = load_dataset("DBD-research-group/BirdSet", "HSN", split="train")
test_ds = load_dataset("DBD-research-group/BirdSet", "HSN", split="test_5s")

# the dataset comes without an automatic Audio casting, this has to be enabled via huggingface
# this means that each time a sample is called, it is decoded (which may take a while if done for the complete dataset)
# in BirdSet, this is all done on-the-fly during training and testing (since the dataset size would be too big if mapping and saving it only once)
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=32_000))
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=32_000))

# extract the first five seconds of each sample in training (not utilizing event detection)
# this is not very efficient since each complete audio file must be decoded this way.
# a custom decoding with soundfile, stating start and end would be more efficient (see BirdSet Code)
def map_first_five(sample):
    max_length = 160_000 # 32_000hz*5sec
    sample["audio"]["array"] =  sample["audio"]["array"][:max_length]
    return sample

# train is now available as an array that can be transformed into a spectrogram for example 
train_ds = train_ds.map(map_first_five, batch_size=1000, num_proc=4)

# the test_5s dataset is already divided into 5-second chunks where each sample can have zero, one or multiple bird vocalizations (ebird_code labels)
test_ds = test_ds.map(map_first_five, batch_size=1000, num_proc=4)

print(train_ds)
print(test_ds)