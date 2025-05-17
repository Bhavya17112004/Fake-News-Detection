import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def enhanced_wordopt(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    return text

def load_and_preprocess_data():
    # Load datasets
    fake_data = pd.read_csv('Datasets/Fake.csv')
    true_data = pd.read_csv('Datasets/True.csv')
    
    # Add labels
    fake_data['label'] = 0  # 0 for fake
    true_data['label'] = 1  # 1 for true
    
    # Combine datasets
    data = pd.concat([fake_data, true_data], axis=0)
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Combine title and text
    data['combined_text'] = data['title'] + ' ' + data['text']
    
    # Clean the text
    data['cleaned_text'] = data['combined_text'].apply(enhanced_wordopt)
    
    return data 

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, early_stopping_patience=5):
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_prec = precision_score(train_targets, train_preds, average='weighted')
        train_rec = recall_score(train_targets, train_preds, average='weighted')
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds, average='weighted')
        val_rec = recall_score(val_targets, val_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_acc)
        metrics['val_accuracy'].append(val_acc)
        metrics['train_precision'].append(train_prec)
        metrics['val_precision'].append(val_prec)
        metrics['train_recall'].append(train_rec)
        metrics['val_recall'].append(val_rec)
        metrics['train_f1'].append(train_f1)
        metrics['val_f1'].append(val_f1)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_model.pth'))
    
    # Plot training curves
    plot_training_curves(metrics)
    
    return model, metrics

def plot_training_curves(metrics):
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy curves
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_accuracy'], label='Train')
    plt.plot(metrics['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # F1 curves
    plt.subplot(2, 2, 3)
    plt.plot(metrics['train_f1'], label='Train')
    plt.plot(metrics['val_f1'], label='Validation')
    plt.title('F1 Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Precision-Recall curves
    plt.subplot(2, 2, 4)
    plt.plot(metrics['train_precision'], label='Train Precision')
    plt.plot(metrics['train_recall'], label='Train Recall')
    plt.plot(metrics['val_precision'], label='Validation Precision')
    plt.plot(metrics['val_recall'], label='Validation Recall')
    plt.title('Precision-Recall Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png')
    plt.close()

def save_model_and_metrics(model, vectorizer, metrics, model_name):
    # Save model
    torch.save(model.state_dict(), f'models/{model_name}_model.pth')
    
    # Save vectorizer
    import joblib
    joblib.dump(vectorizer, f'models/{model_name}_vectorizer.pkl')
    
    # Save metrics
    import json
    with open(f'models/{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f) 