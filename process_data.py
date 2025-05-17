import pandas as pd
import os

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)

# Load the raw data
true_df = pd.read_csv('Datasets/True.csv')
fake_df = pd.read_csv('Datasets/Fake.csv')

# Add labels
true_df['label'] = 1  # 1 for true news
fake_df['label'] = 0  # 0 for fake news

# Combine the datasets
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the data
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the processed data
combined_df.to_csv('data/processed/train.csv', index=False)

print("Data processing completed. Processed data saved to data/processed/train.csv") 