# Phishing Detector - BERT + LSTM Deep Learning UI

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org)

Production-ready phishing email detector with **BERT Transformer + Bidirectional LSTM** deep learning models. Features live UI with model agreement, confidence comparison, final verdict, risk indicator, and processing time.

## 🎯 Live Demo Features
```
🤖 BERT Prediction 
🧠 LSTM Prediction 
📊 Model Agreement Status (✅ Agree / ⚠️ Disagree)
📈 Average Confidence Score
🎯 Final Combined Verdict
🟢 Risk Level Indicator
```

## 🚀 Quick Start (Local)

### 1. Clone & Install
```bash
git clone https://github.com/kasmya/phising-detector-dl.git
cd phising-detector-dl
pip install -r requirements.txt
```

### 2. Run Server
```bash
python app.py
```
**Open:** http://localhost:5001

### 3. Test Analysis
**Legitimate Invoice:**
```
Invoice #1234
Dear Customer,
Your invoice #1234 is ready. Download: invoice1234.pdf
Questions? support@company.com
Thanks, Billing Dept
```
**Expected:** `LEGITIMATE ✅ | Low Risk | Models Agree`

**Phishing Alert:**
```
URGENT: Your account is LOCKED! Verify immediately:
http://fake-bank-security.com/verify-account
24hrs or PERMANENT loss!
```
**Expected:** `PHISHING 🔴 | High Risk | Models Agree`

## 🤖 Model Details

### Model Training
- **BERT:** Fine-tuned on phishing emails (HuggingFace)
- **LSTM:** Custom bidirectional RNN with class weights
- **Vocab:** 8K tokens (`lstm_vocab_fixed.json`)
- **Max Len:** 80 tokens

## 📁 Project Structure
```
phising-detector-dl/
├── app.py                 # Flask backend + models
├── templates/index.html   # Responsive UI
├── requirements.txt       # Dependencies
├── lstm_phishing_fixed.pth # LSTM model (91% acc)
├── lstm_vocab_fixed.json # LSTM vocab
├── phishing_model/        # BERT model files
├── README.md             # This file
```


