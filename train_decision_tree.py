import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from model_architectures import DecisionTreeClassifier
from utils import TextDataset, train_model, save_model_and_metrics

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
print("Loading data...")
df = pd.read_csv('data/processed/train.csv')
X = df['text']
y = df['label']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_vec.toarray())
y_train_tensor = torch.LongTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val_vec.toarray())
y_val_tensor = torch.LongTensor(y_val.values)

# Create datasets and dataloaders
train_dataset = TextDataset(X_train_tensor, y_train_tensor)
val_dataset = TextDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model with regularization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DecisionTreeClassifier(
    input_dim=5000,
    hidden_dim=256,
    num_classes=2,
    dropout_rate=0.3
).to(device)

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Train model
print("Training model...")
model, metrics = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=20,
    early_stopping_patience=5
)

# Generate predictions for confusion matrix
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor.to(device))
    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

# Generate confusion matrix
cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('models/decision_tree_confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=['Fake', 'Real']))

# Save model and metrics
save_model_and_metrics(
    model=model,
    vectorizer=vectorizer,
    metrics=metrics,
    model_name='decision_tree'
)

print("Training completed!") 