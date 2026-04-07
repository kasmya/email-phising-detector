from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import re
import nltk
from nltk.corpus import stopwords
from src.preprocess import TextPreprocessor
from src.models.bert_model import BERTModel
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os

app = Flask(__name__)

# Global instances loaded at startup
preprocessor = TextPreprocessor()
bert_model = BERTModel()
bert_model.load_saved_model('./phishing_model')
bert_model.model.eval()

# Load NEW TinyLSTM from flask_model/
model_path = 'lstm_phishing_fixed.pth'
vocab_path = 'flask_model/vocab.json'

# TinyLSTM class from new_lstm.ipynb
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.lstm.bidirectional else hidden[-1]
        out = self.dropout(hidden_last)
        out = self.fc(out)
        return out.squeeze(1)

# Load fixed LSTM model
lstm_state_dict = torch.load(model_path, map_location='cpu')
lstm_model = LSTMClassifier(
    vocab_size=8000,
    embedding_dim=64,
    hidden_dim=64,
    num_layers=2,
    num_classes=2,
    bidirectional=True
)
lstm_model.load_state_dict(lstm_state_dict['model_state_dict'])
lstm_model.eval()
print("Fixed LSTM model loaded from lstm_phishing_fixed.pth!")

def simple_preprocess(text):
    """From new_lstm.ipynb"""
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if len(word) > 3 and word not in stop_words]
    return tokens[:50]

def to_sequence(tokens, word_to_idx, max_len=30):
    seq = [word_to_idx.get(t, 1) for t in tokens]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return seq

def predict_fixed_lstm(text):
    try:
        sequence = preprocessor.prepare_for_lstm(text)
        input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logit = lstm_model(input_tensor)
            prob = F.sigmoid(logit).item()
        return {
            'is_phishing': prob > 0.5,
            'phishing_probability': prob,
            'confidence': max(prob, 1-prob)
        }
    except Exception as e:
        print(f"LSTM error: {e}")
        return {'is_phishing': False, 'phishing_probability': 0.5, 'confidence': 0.5}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Analytics
    analytics = preprocessor.extract_features(text)
    
# Start timing
    start_time = time.perf_counter()
    
    # BERT prediction
    start_time = time.perf_counter()
    
    # BERT prediction
    bert_pred = bert_model.predict_single(text)
    
    # Fixed LSTM prediction
    lstm_pred = predict_fixed_lstm(text)
    
    # End timing
    analysis_time = time.perf_counter() - start_time
    
    # Compute summary
    bert_is_phish = bert_pred.get('is_phishing', False)
    lstm_is_phish = lstm_pred['is_phishing']
    
    agreement = "✅ Agree" if bert_is_phish == lstm_is_phish else "⚠️ Disagree"
    agreement_class = "agree" if bert_is_phish == lstm_is_phish else "disagree"
    
    bert_conf = bert_pred.get('confidence', 0.5)
    lstm_conf = lstm_pred['confidence']
    avg_confidence = (bert_conf + lstm_conf) / 2
    
    final_verdict = "LEGITIMATE" if (bert_is_phish and lstm_is_phish) == False else "PHISHING"
    if agreement == "⚠️ Disagree":
        final_verdict = "UNCERTAIN"  # When models disagree
    
    risk_level = "🟢 Low Risk" if final_verdict == "LEGITIMATE" else "🔴 High Risk" if final_verdict == "PHISHING" else "🟡 Medium Risk"
    
    summary = {
        'models_agreement': agreement,
        'agreement_class': agreement_class,
        'avg_confidence': avg_confidence,
        'final_verdict': final_verdict,
        'analysis_time': f"{analysis_time:.2f}s",
        'risk_level': risk_level,
        'models_analyzed': 2
    }
    
    return jsonify({
        'analytics': analytics,
        'bert': bert_pred,
    'lstm': lstm_pred,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
