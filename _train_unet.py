import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys

# Import custom modules
sys.path.append('.')
from src.models.unet import UNet
from src.evaluation.metrics import SegmentationMetrics, DiceBCELoss
from src.data.dataloader import create_dataloaders

print("="*60)
print("TRAINING U-NET ON IDC DATASET")
print("="*60)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate metrics
            preds = torch.sigmoid(outputs)
            metrics = SegmentationMetrics.calculate_all_metrics(preds, labels)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    val_loss = running_loss / len(val_loader)
    return val_loss, avg_metrics

def train_model(config):
    """Main training function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config['dataset_path'],
        splits_path=config['splits_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=config['image_size']
    )
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, 
                                  factor=0.5, verbose=True)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = np.Inf
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, 
                                     optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}")
        print(f"  Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Val F1: {val_metrics['f1_score']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, 'models/checkpoints/best_unet.pth')
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save training history
    with open('results/unet/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # IoU
    iou_scores = [m['iou'] for m in history['val_metrics']]
    axes[0, 1].plot(epochs, iou_scores, label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU Score')
    axes[0, 1].set_title('Validation IoU Score')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Dice
    dice_scores = [m['dice'] for m in history['val_metrics']]
    axes[1, 0].plot(epochs, dice_scores, label='Val Dice', color='blue')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Multiple metrics
    f1_scores = [m['f1_score'] for m in history['val_metrics']]
    precision_scores = [m['precision'] for m in history['val_metrics']]
    recall_scores = [m['recall'] for m in history['val_metrics']]
    
    axes[1, 1].plot(epochs, f1_scores, label='F1 Score')
    axes[1, 1].plot(epochs, precision_scores, label='Precision')
    axes[1, 1].plot(epochs, recall_scores, label='Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/unet/training_curves.png', dpi=300)
    print(f"\n✅ Saved training curves to: results/unet/training_curves.png")

if __name__ == "__main__":
    # Training configuration
    config = {
        'dataset_path': r'D:\projs\segment_anything\data\IDC_regular_ps50_idx5',
        'splits_path': 'dataset_splits.json',
        'batch_size': 32,
        'num_workers': 4,
        'image_size': 256,
        'learning_rate': 0.001,
        'epochs': 50,
        'patience': 7
    }
    
    # Train model
    model, history = train_model(config)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print("="*60)
