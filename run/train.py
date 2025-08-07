"""
Training module for the CNN lip-reading model.

This module provides functions for training the CNN model on lip-reading data,
including dataset loading, model training, and performance visualization.
"""

import os
import os.path
from time import time
from typing import Tuple, List, Optional

import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

from constants.constants import CLASSES, MODEL_PATH
from infra.cnn import CNN
from utils.create_dataset import create_datasets
from utils.general_utils import get_encoded_labels, convert_tuple
from utils.plots import plot_train_loss, plot_train_accuracy


def train(num_epochs: int = 40, train_ratio: float = 0.7, 
          learning_rate: float = 0.001, batch_size: int = 1) -> CNN:
    """
    Train the CNN model on the lip-reading dataset.
    
    This function handles the complete training pipeline including data loading,
    model initialization, training loop, and performance tracking.
    
    Args:
        num_epochs (int): Number of training epochs
        train_ratio (float): Ratio of data to use for training (0.0-1.0)
        learning_rate (float): Learning rate for the optimizer
        batch_size (int): Batch size for training
        
    Returns:
        CNN: Trained model
        
    Raises:
        ValueError: If train_ratio is not between 0 and 1
        RuntimeError: If no training data is available
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    
    print(f"🚀 Starting training with {num_epochs} epochs, train_ratio={train_ratio}")
    print(f"📊 Learning rate: {learning_rate}, Batch size: {batch_size}")
    
    # Load datasets
    try:
        (train_dataset, test_dataset, dataset), (num_samples, num_train_samples) = create_datasets(train_ratio)
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {e}")
    
    if num_train_samples == 0:
        raise RuntimeError("No training samples available")
    
    print(f"📈 Dataset loaded: {num_samples} total samples, {num_train_samples} for training")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Initialize model
    model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set model to training mode
    model.train()

    print(f"🎯 Training {len(CLASSES)} classes: {CLASSES}")
    print(f"⏱️  Expected training samples per epoch: {num_train_samples}")
    print("=" * 60)

    # Training metrics
    start_time = time()
    train_loss_history: List[float] = []
    train_acc_history: List[float] = []
    
    try:
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            print(f"\n📚 Epoch [{epoch + 1}/{num_epochs}]")
            
            for i, (input_data, class_label) in enumerate(train_loader):
                try:
                    # Encode labels
                    encoded_labels = get_encoded_labels()
                    label_key = convert_tuple(class_label)
                    
                    if label_key not in encoded_labels:
                        print(f"⚠️  Warning: Unknown class {label_key}, skipping sample")
                        continue
                    
                    label = torch.tensor([encoded_labels[label_key]], dtype=torch.long)
                    
                    # Move data to device
                    input_data = input_data.to(device)
                    label = label.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(input_data)
                    loss = criterion(outputs, label)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    epoch_total += label.size(0)
                    epoch_correct += predicted.eq(label).sum().item()

                    # Print progress
                    if (i + 1) % 100 == 0:
                        current_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0
                        print(f'  Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
                        
                except Exception as e:
                    print(f"⚠️  Error processing batch {i}: {e}")
                    continue

            # Calculate epoch metrics
            if len(train_loader) > 0:
                epoch_loss /= len(train_loader)
                epoch_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0
                
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                
                # Print epoch summary
                elapsed_time = time() - start_time
                print(f"✅ Epoch {epoch + 1} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {elapsed_time:.1f}s")
            
        total_time = time() - start_time
        print("=" * 60)
        print(f"🎉 Training completed in {total_time:.2f} seconds")
        print(f"📊 Final metrics - Loss: {train_loss_history[-1]:.4f}, Accuracy: {train_acc_history[-1]:.2f}%")
        
        # Plot training curves
        print("📈 Generating training plots...")
        plot_train_loss(train_loss_history)
        plot_train_accuracy(train_acc_history)
        
        return model
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return model
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def save_model(model: CNN, filepath: Optional[str] = None) -> str:
    """
    Save the trained model to disk.
    
    Args:
        model (CNN): Trained model to save
        filepath (Optional[str]): Path to save the model, defaults to MODEL_PATH
        
    Returns:
        str: Path where the model was saved
        
    Raises:
        OSError: If the model cannot be saved
    """
    if filepath is None:
        # Ensure model directory exists
        os.makedirs(MODEL_PATH, exist_ok=True)
        filepath = os.path.join(MODEL_PATH, "cnn_model.pth")
    
    try:
        torch.save(model.state_dict(), filepath)
        print(f"💾 Model saved successfully to: {filepath}")
        
        # Save model metadata
        metadata = {
            'num_classes': len(CLASSES),
            'classes': CLASSES,
            'model_architecture': 'CNN',
            'input_shape': (60, 4),
            'save_timestamp': time()
        }
        
        metadata_path = filepath.replace('.pth', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📝 Model metadata saved to: {metadata_path}")
        return filepath
        
    except Exception as e:
        raise OSError(f"Failed to save model: {e}")


def main() -> None:
    """Main training execution function."""
    try:
        # Get training parameters from user
        epochs = int(input("How many epochs? "))
        
        # Optional: get other parameters
        use_defaults = input("Use default parameters? (y/n): ").lower().strip() == 'y'
        
        if use_defaults:
            train_ratio = 0.7
            learning_rate = 0.001
        else:
            train_ratio = float(input("Train ratio (0.7): ") or "0.7")
            learning_rate = float(input("Learning rate (0.001): ") or "0.001")
        
        print(f"🎯 Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Train ratio: {train_ratio}")
        print(f"  Learning rate: {learning_rate}")
        
        # Train the model
        model = train(
            num_epochs=epochs,
            train_ratio=train_ratio,
            learning_rate=learning_rate
        )
        
        # Save the model
        model_path = save_model(model)
        
        print(f"\n✅ Training complete! Model saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("\n👋 Training cancelled by user")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()
