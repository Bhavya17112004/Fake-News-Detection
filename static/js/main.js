document.addEventListener('DOMContentLoaded', function() {
    const newsSelect = document.getElementById('news-select');
    const newsText = document.getElementById('newsText');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultSection = document.querySelector('.result-section');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const modelSelect = document.getElementById('model-select');
    
    // Metrics elements
    const accuracyMetric = document.getElementById('accuracy-metric');
    const precisionMetric = document.getElementById('precision-metric');
    const recallMetric = document.getElementById('recall-metric');
    const f1Metric = document.getElementById('f1-metric');

    // Model metrics data
    const modelMetrics = {
        'decision_tree_model.pkl': {
            'accuracy': 0.98,
            'precision': 0.97,
            'recall': 0.98,
            'f1_score': 0.98
        },
        'gradient_boosting_model.pkl': {
            'accuracy': 0.99,
            'precision': 0.99,
            'recall': 0.99,
            'f1_score': 0.99
        },
        'random_forest_model.pkl': {
            'accuracy': 0.99,
            'precision': 0.99,
            'recall': 0.99,
            'f1_score': 0.99
        },
        'logistic_model.pkl': {
            'accuracy': 0.98,
            'precision': 0.98,
            'recall': 0.98,
            'f1_score': 0.98
        }
    };

    // Update metrics when model is selected
    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        const metrics = modelMetrics[selectedModel];
        
        accuracyMetric.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
        precisionMetric.textContent = `${(metrics.precision * 100).toFixed(1)}%`;
        recallMetric.textContent = `${(metrics.recall * 100).toFixed(1)}%`;
        f1Metric.textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;
    });

    // Update textarea when news is selected
    newsSelect.addEventListener('change', function() {
        newsText.value = this.value;
    });

    // Handle analyze button click
    analyzeBtn.addEventListener('click', async function() {
        const text = newsText.value.trim();
        if (!text) {
            alert('Please enter or select a news article to analyze.');
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: modelSelect.value
                })
            });

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Update prediction result
            predictionResult.textContent = data.prediction;
            confidenceResult.textContent = `${data.confidence}%`;
            
            // Show result section
            resultSection.style.display = 'block';
            
            // Update metrics
            const metrics = data.metrics;
            accuracyMetric.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
            precisionMetric.textContent = `${(metrics.precision * 100).toFixed(1)}%`;
            recallMetric.textContent = `${(metrics.recall * 100).toFixed(1)}%`;
            f1Metric.textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the news. Please try again.');
        }
    });

    // Initialize metrics for default model
    modelSelect.dispatchEvent(new Event('change'));
}); 