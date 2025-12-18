# train_emotion_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import json
from pathlib import Path
from utils.model_predictor import AdvancedEmotionModel

class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze()

def create_sample_dataset():
    """Create a sample dataset for training"""
    print("🔄 Creating sample dataset...")
    
    # Sample data structure
    features = []
    labels = []
    
    # Generate synthetic audio features for each emotion
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    
    for emotion_idx, emotion in enumerate(emotions):
        for i in range(100):  # 100 samples per emotion
            # Generate features with emotion-specific patterns
            sample_features = np.random.normal(0, 1, 158)
            
            # Add emotion-specific biases
            if emotion == "happy":
                sample_features[13:26] += np.random.normal(0.5, 0.2, 13)  # Higher spectral features
            elif emotion == "sad":
                sample_features[13:26] += np.random.normal(-0.5, 0.2, 13)  # Lower spectral features
            elif emotion == "angry":
                sample_features[24:30] += np.random.normal(0.8, 0.3, 6)   # Higher ZCR/RMS
            elif emotion == "calm":
                sample_features[24:30] += np.random.normal(-0.3, 0.1, 6)  # Lower ZCR/RMS
            
            features.append(sample_features)
            labels.append(emotion_idx)
    
    return np.array(features), np.array(labels)

def train_model():
    """Train the emotion recognition model"""
    print("🚀 Starting model training...")
    
    # Configuration
    config = {
        "input_size": 158,
        "num_classes": 8,
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 50
    }
    
    # Create sample dataset
    features, labels = create_sample_dataset()
    
    # Split dataset
    split_idx = int(0.8 * len(features))
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create data loaders
    train_dataset = EmotionDataset(train_features, train_labels)
    val_dataset = EmotionDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedEmotionModel(config["input_size"], config["num_classes"]).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = 100 * correct / total
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{config["epochs"]}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'config': config
            }, 'assets/models/best_emotion_model.pth')
            print(f'✅ Best model saved with accuracy: {best_accuracy:.2f}%')
    
    print(f'🎉 Training completed! Best accuracy: {best_accuracy:.2f}%')
    
    # Save config
    os.makedirs('assets/models', exist_ok=True)
    with open('assets/models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return model, best_accuracy

if __name__ == "__main__":
    train_model()