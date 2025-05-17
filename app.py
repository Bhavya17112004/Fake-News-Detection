from flask import Flask, render_template, request, jsonify
import torch
import joblib
import numpy as np
from model_architectures import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    LSTMModel,
    GradientBoostingClassifier
)

app = Flask(__name__)

# Load models and vectorizers
def load_model(model_name):
    model_path = f'models/{model_name}_model.pth'
    vectorizer_path = f'models/{model_name}_vectorizer.pkl'
    
    # Load vectorizer
    vectorizer = joblib.load(vectorizer_path)
    
    # Initialize and load model
    if model_name == 'logistic_regression':
        model = LogisticRegression(input_dim=5000, num_classes=2)
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(input_dim=5000, hidden_dim=256, num_classes=2)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(input_dim=5000, hidden_dim=512, num_classes=2)
    elif model_name == 'lstm':
        model = LSTMModel(input_dim=5000, hidden_dim=256, num_layers=2, num_classes=2)
    elif model_name == 'gradient_boosting':
        model = GradientBoostingClassifier(input_dim=5000, hidden_dim=512, num_classes=2)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, vectorizer

# Load all models
models = {
    'logistic_regression': load_model('logistic_regression'),
    'decision_tree': load_model('decision_tree'),
    'random_forest': load_model('random_forest'),
    'lstm': load_model('lstm'),
    'gradient_boosting': load_model('gradient_boosting')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        selected_models = data.get('models', [])  # Get selected models from frontend
        
        if not selected_models:
            return jsonify({'error': 'Please select at least one model'}), 400
        
        predictions = {}
        for model_name in selected_models:
            if model_name in models:
                model, vectorizer = models[model_name]
                
                # Vectorize text
                X = vectorizer.transform([text])
                X = torch.FloatTensor(X.toarray())
                
                # Get prediction
                with torch.no_grad():
                    output = model(X)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                predictions[model_name] = {
                    'prediction': 'Real' if prediction == 1 else 'Fake',
                    'confidence': f'{confidence:.2%}'
                }
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 