import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

from sincnet_io import AudioDataset, create_dataloaders
from SincNetModel import SincNetModel, SincNetConfig

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

# Set up data loaders
train_loader, val_loader = create_dataloaders(
    root_dir="nips4bplus",
    batch_size=256,
    sample_rate=cfg.sample_rate,
    cw_len=cfg.cw_len,
    augment_factor=0
)

# Print dataset sizes and number of batches
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of batches per epoch: {len(train_loader)}")

# Print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters in the model: {total_params:,}")
print(f"Number of trainable parameters: {trainable_params:,}")
print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")

# import sys; sys.exit(0)

criterion = nn.CrossEntropyLoss() # same as NLLLoss originally used in SincNet, but more efficient and stable
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

num_epochs = 200
log_file = os.path.join(os.path.dirname(__file__), "trainlog.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    epoch_start_time = time.time()
    for batch_idx, batch in enumerate(train_loader, 1):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # log train loss per step
        with open(log_file, "a") as f:
            f.write(f"{epoch * len(train_loader) + batch_idx} train {loss.item():.4f}\n")
    
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        class_correct = torch.zeros(cfg.num_classes, device=device)
        class_total = torch.zeros(cfg.num_classes, device=device)
        class_predicted = torch.zeros(cfg.num_classes, device=device)
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for c in range(cfg.num_classes):
                class_mask = labels == c
                class_correct[c] += (predicted[class_mask] == c).sum()
                class_total[c] += class_mask.sum()
                class_predicted[c] += (predicted == c).sum()

        # Handle division by zero
        precision = torch.where(class_predicted != 0, class_correct / class_predicted, torch.zeros_like(class_predicted))
        recall = torch.where(class_total != 0, class_correct / class_total, torch.zeros_like(class_total))
        
        # Calculate mean precision and recall, ignoring NaN values
        precision_mean = precision[~torch.isnan(precision)].mean().item()
        recall_mean = recall[~torch.isnan(recall)].mean().item()
        
        # Calculate F1 score, avoiding division by zero
        f1 = torch.where(
            (precision + recall) != 0,
            2 * (precision * recall) / (precision + recall),
            torch.zeros_like(precision)
        ).mean().item()

        accuracy = correct / total if total > 0 else 0

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} |"
              f"Train Loss: {avg_train_loss:.4f} |"
              f"Val Loss: {val_loss/len(val_loader):.4f} |"
              f"Val Acc: {accuracy:.4f} |"
              f"Precision: {precision_mean:.4f} |"
              f"Recall: {recall_mean:.4f} |"
              f"F1: {f1:.4f} |"
              f"Time: {epoch_duration:.2f} seconds")
        
        # log epoch validation metrics
        with open(log_file, "a") as f:
            f.write(f"{(epoch + 1) * len(train_loader)} val {val_loss/len(val_loader):.4f} "
                    f"{accuracy:.4f} {precision_mean:.4f} {recall_mean:.4f} {f1:.4f}\n")


import sys; sys.exit(0)
# Get the first batch from the train loader
first_batch, first_labels = next(iter(train_loader))

# Move the batch to the device
first_batch = first_batch.to(device)

# Set the model to evaluation mode
model.eval()

# Compute logits for the first example
with torch.no_grad():
    first_example = first_batch[0].unsqueeze(0)  # Add batch dimension
    #print(first_example)
    logits = model(first_example)

print(f"Logits for the first example: {logits}")
print(f"Shape of logits: {logits.shape}")
