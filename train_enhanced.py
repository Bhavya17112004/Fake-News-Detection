import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

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

def train_models():
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], 
        data['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Initialize vectorizer with enhanced parameters
    vectorizer = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 2),  # Include both single words and pairs of words
        min_df=5,  # Minimum document frequency
        max_df=0.7  # Maximum document frequency
    )
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save the vectorizer
    joblib.dump(vectorizer, 'enhanced_tfidf_vectorizer.pkl')
    
    # Define models with optimized parameters
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=50,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Save the model
        model_filename = f'enhanced_{name}_model.pkl'
        joblib.dump(model, model_filename)
        
        print(f"{name} metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Save results
    joblib.dump(results, 'enhanced_model_metrics.pkl')
    
    return results

if __name__ == '__main__':
    print("Starting enhanced model training...")
    results = train_models()
    print("\nTraining completed!")
    print("\nFinal Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}") 