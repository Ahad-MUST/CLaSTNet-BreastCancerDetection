import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import train_dataset, val_dataset, test_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
from CLaST import CLaST
from evaluate import Evaluator

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect predictions and labels for metrics calculation
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validates the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Collect predictions and labels for metrics calculation
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    return avg_loss, accuracy

def main():
    """Main function to train, validate, and evaluate the model."""
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    model = CLaST(img_size=224, patch_size=2, num_classes=2,
                 embed_dim=36, conv_depths=2, depths=[5, 3], num_heads=2,
                 window_size=[5, 5, 5], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 multi_shift=False, shift_window=True)
    model.to(device)
    print(f'Starting training on {device}...')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training and validation loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch", leave=False)
        train_loss, train_accuracy = train_one_epoch(model, train_loader_tqdm, criterion, optimizer, device)
        
        # Validation
        val_loader_tqdm = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}", unit="batch", leave=False)
        val_loss, val_accuracy = validate(model, val_loader_tqdm, criterion, device)

        # Calculate epoch duration
        epoch_duration = time.time() - start_time

        # Update learning rate scheduler
        scheduler.step()

        # Display epoch results
        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% "
              f"(Duration: {epoch_duration:.2f}s)")

    # Evaluate on test data
    print("\nEvaluating on Test Data...")
    evaluator = Evaluator(model, test_loader, input_size=(1, 3, 224, 224), device=device)
    evaluator.analyze_model()
    evaluator.evaluate()
    evaluator.show_predictions(num_samples=8)

if __name__ == '__main__':
    main()
