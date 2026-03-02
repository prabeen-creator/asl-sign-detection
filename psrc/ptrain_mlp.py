#!/usr/bin/env python
"""
Train a Multi-Layer Perceptron (MLP) for ASL sign language classification.
Includes Weights and Biases integration for experiment tracking.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Experiment tracking disabled.")


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for ASL classification."""
    
    def __init__(self, input_size=42, hidden_sizes=[128, 64], num_classes=24, 
                 activation='relu', dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        
        layers = []
        prev_size = input_size
        
        activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }.get(activation, nn.ReLU())
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data(csv_path):
    """Load and preprocess the landmark data."""
    df = pd.read_csv(csv_path)
    
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder


def create_dataloaders(X, y, batch_size=32, test_size=0.2):
    """Create train and test dataloaders."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_test, y_test


def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=50, device='cpu', use_wandb=False):
    """Train the MLP model."""
    model = model.to(device)
    best_accuracy = 0.0
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_accuracy'].append(accuracy)
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'test_accuracy': accuracy
            })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
    
    return history, best_accuracy


def evaluate_model(model, X_test, y_test, label_encoder, device='cpu'):
    """Evaluate the model and print classification report."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, 
        target_names=label_encoder.classes_
    ))
    
    return predictions


def save_model(model, label_encoder, hidden_sizes, num_classes, save_path):
    """Save the trained model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'hidden_sizes': hidden_sizes,
        'num_classes': num_classes,
        'input_size': 42
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train MLP for ASL classification')
    parser.add_argument('--data', type=str, default='asl_landmarks.csv',
                        help='Path to landmarks CSV file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer sizes (e.g., --hidden 128 64)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh', 'leaky_relu', 'elu'],
                        help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights and Biases logging')
    parser.add_argument('--wandb-project', type=str, default='asl-sign-detection',
                        help='Weights and Biases project name')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            config={
                'model': 'MLP',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'hidden_sizes': args.hidden,
                'activation': args.activation,
                'dropout': args.dropout
            }
        )
    
    print("Loading data...")
    X, y, label_encoder = load_data(args.data)
    num_classes = len(label_encoder.classes_)
    print(f"Loaded {len(X)} samples with {num_classes} classes")
    
    train_loader, test_loader, X_test, y_test = create_dataloaders(
        X, y, batch_size=args.batch_size
    )
    
    model = MLPClassifier(
        input_size=42,
        hidden_sizes=args.hidden,
        num_classes=num_classes,
        activation=args.activation,
        dropout=args.dropout
    )
    print(f"\nModel architecture:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nTraining...")
    history, best_accuracy = train_model(
        model, train_loader, test_loader,
        criterion, optimizer,
        num_epochs=args.epochs,
        device=device,
        use_wandb=args.wandb and WANDB_AVAILABLE
    )
    
    evaluate_model(model, X_test, y_test, label_encoder, device)
    
    save_model(
        model, label_encoder, args.hidden, num_classes,
        'models/asl_mlp_model.pt'
    )
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
