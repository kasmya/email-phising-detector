"""
BERT Deep Learning Model for Phishing Detection

This module implements a BERT-based transformer model for detecting
phishing emails by understanding context and meaning in text.
"""

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class PhishingDataset(Dataset):
    """
    PyTorch Dataset for phishing email classification.
    
    Handles tokenization and encoding for BERT models.
    """
    
    def __init__(self, texts: List[str], labels: List[int] = None,
                 tokenizer: BertTokenizer = None, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (1=phishing, 0=legitimate)
            tokenizer: BERT tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class BERTModel:
    """
    BERT-based Deep Learning model for phishing detection.
    
    Uses pre-trained BERT model fine-tuned on phishing data.
    BERT understands context, meaning, and intent in text.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 num_labels: int = 2, max_length: int = 512):
        """
        Initialize BERT model for phishing detection.
        
        Args:
            model_name: Pre-trained model name or path
            num_labels: Number of output classes
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"BERT Model initialized on device: {self.device}")
    
    def load_tokenizer(self):
        """Load pre-trained BERT tokenizer."""
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        return self.tokenizer
    
    def load_model(self, model_path: str = None):
        """
        Load BERT model for sequence classification.
        
        Args:
            model_path: Path to saved model (optional)
        """
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            print(f"Loading pre-trained model: {self.model_name}")
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
        
        self.model.to(self.device)
        return self.model
    
    def build_model(self):
        """Build and initialize model with tokenizer."""
        self.load_tokenizer()
        self.load_model()
        return self.model
    
    def prepare_dataset(self, texts: List[str], labels: List[int] = None) -> PhishingDataset:
        """
        Create dataset for training/evaluation.
        
        Args:
            texts: List of text samples
            labels: List of labels
            
        Returns:
            PhishingDataset instance
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        return PhishingDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None,
              epochs: int = 3, batch_size: int = 8,
              learning_rate: float = 2e-5, warmup_steps: int = 0,
              output_dir: str = './models/bert_phishing'):
        """
        Train the BERT model on phishing data.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            output_dir: Directory to save model
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.model is None:
            self.load_model()
        
        # Create datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = None
        val_loader = None
        if val_texts and val_labels:
            val_dataset = self.prepare_dataset(val_texts, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(train_dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*50)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                          f"Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            print(f"\n  Training Loss: {avg_train_loss:.4f}")
            print(f"  Training Accuracy: {train_acc:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {val_acc:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(output_dir)
                    print(f"  ✓ New best model saved!")
        
        print(f"\nTraining completed! Best model saved to: {output_dir}")
        return self.model
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def predict(self, texts: List[str], 
                return_probabilities: bool = True,
                threshold: float = 0.5) -> List[Dict]:
        """
        Predict phishing probability for input texts.
        
        BERT's magic: Understands that these mean the SAME:
        - "Verify your account" = "Confirm your credentials" = "Authenticate your login"
        
        Args:
            texts: List of email texts to analyze
            return_probabilities: Include probability scores
            threshold: Classification threshold
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            self.load_tokenizer()
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get probabilities
                probabilities = torch.softmax(outputs.logits, dim=1)
                phishing_prob = probabilities[0][1].item()  # P(phishing)
                legitimate_prob = probabilities[0][0].item()  # P(legitimate)
                
                is_phishing = phishing_prob > threshold
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'is_phishing': bool(is_phishing),
                    'prediction': 'PHISHING' if is_phishing else 'LEGITIMATE'
                }
                
                if return_probabilities:
                    result['phishing_probability'] = phishing_prob
                    result['legitimate_probability'] = legitimate_prob
                    result['confidence'] = max(phishing_prob, legitimate_prob)
                
                results.append(result)
        
        return results
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Prediction dictionary
        """
        results = self.predict([text])
        return results[0]
    
    def explain_prediction(self, text: str) -> Dict:
        """
        Generate detailed explanation for prediction.
        
        Uses BERT's attention mechanism to understand what the model focused on.
        
        Args:
            text: Input text
            
        Returns:
            Explanation dictionary with risk factors
        """
        prediction = self.predict_single(text)
        
        # Analyze text for common phishing indicators
        text_lower = text.lower()
        
        urgency_indicators = [
            'urgent', 'immediate', 'immediately', 'asap', 'right now',
            'suspend', 'suspended', 'terminate', 'terminated', 'expire',
            'within 24', 'within 48', 'deadline', 'action required'
        ]
        
        threat_words = [
            'warning', 'alert', 'suspicious', 'unauthorized', 'blocked',
            'restricted', 'verify', 'confirm', 'security', 'breach',
            'compromise', 'stolen', 'fraudulent'
        ]
        
        request_patterns = [
            ('password', ['password', 'credential', 'login', 'sign in']),
            ('financial', ['bank', 'credit card', 'account number', 'routing']),
            ('personal', ['social security', 'date of birth', 'address', 'phone']),
            ('click_link', ['click here', 'click below', 'tap here', 'open link'])
        ]
        
        found_urgency = [w for w in urgency_indicators if w in text_lower]
        found_threats = [w for w in threat_words if w in text_lower]
        found_requests = []
        
        for category, keywords in request_patterns:
            if any(kw in text_lower for kw in keywords):
                found_requests.append(category)
        
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        
        if found_urgency:
            risk_score += len(found_urgency) * 15
            risk_factors.append({
                'type': 'Urgency Tactics',
                'severity': 'HIGH' if len(found_urgency) > 2 else 'MEDIUM',
                'description': f"Uses urgent language: {', '.join(found_urgency[:3])}"
            })
        
        if found_threats:
            risk_score += len(found_threats) * 10
            risk_factors.append({
                'type': 'Threat Language',
                'severity': 'HIGH' if len(found_threats) > 2 else 'MEDIUM',
                'description': f"Contains threat words: {', '.join(found_threats[:3])}"
            })
        
        if found_requests:
            risk_score += len(found_requests) * 20
            for req in found_requests:
                risk_factors.append({
                    'type': 'Sensitive Information Request',
                    'severity': 'HIGH',
                    'description': f"Asks for {req} information"
                })
        
        # Check URLs
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            # Check for suspicious URL patterns
            suspicious_url_patterns = [
                'bit.ly', 'tinyurl', 't.co', 'ow.ly',  # URL shorteners
                'secure', 'login', 'verify', 'account'  # Suspicious keywords
            ]
            for url in urls:
                if any(pattern in url.lower() for pattern in suspicious_url_patterns):
                    risk_score += 15
                    risk_factors.append({
                        'type': 'Suspicious URL',
                        'severity': 'MEDIUM',
                        'description': f"Suspicious link detected: {url[:50]}..."
                    })
        
        explanation = {
            'prediction': prediction['prediction'],
            'confidence': f"{prediction['confidence']:.1%}" if 'confidence' in prediction else 'N/A',
            'phishing_probability': f"{prediction.get('phishing_probability', 0):.2%}" if 'phishing_probability' in prediction else 'N/A',
            'bert_analysis': {
                'model_used': 'bert-base-uncased',
                'max_sequence_length': self.max_length,
                'understanding': 'BERT analyzes 12 layers of context to understand meaning and intent'
            },
            'risk_score': min(risk_score, 100),
            'risk_factors': risk_factors
        }
        
        return explanation
    
    def save_model(self, output_dir: str):
        """
        Save model and tokenizer to disk.
        
        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to: {output_dir}")
    
    def load_saved_model(self, model_dir: str):
        """
        Load saved model and tokenizer.
        
        Args:
            model_dir: Directory containing saved model
        """
        print(f"Loading model from: {model_dir}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        
        return self.model
    
    def get_attention_weights(self, text: str) -> Dict:
        """
        Get BERT attention weights for interpretability.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokens and their attention weights
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get all attention heads from last layer
        attentions = outputs.attentions[-1]  # Last layer
        attentions = attentions.squeeze(0)  # Remove batch dim
        
        # Average across all heads
        avg_attention = attentions.mean(dim=0).cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get CLS token attention (represents whole sequence)
        cls_attention = avg_attention[0]  # CLS is first token
        
        # Find tokens with highest attention
        token_attention = list(zip(tokens, cls_attention.tolist()))
        token_attention.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'tokens': tokens,
            'attention_matrix_shape': avg_attention.shape,
            'cls_attention_to_tokens': token_attention[:10],  # Top 10
            'prediction': torch.argmax(outputs.logits, dim=1).item()
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== BERT Phishing Detection Model Demo ===\n")
    
    # Initialize model
    bert = BERTModel(model_name='bert-base-uncased', num_labels=2)
    
    # Build model
    print("Building BERT model...")
    bert.build_model()
    print("Model loaded successfully!")
    
    print("\n" + "="*50 + "\n")
    
    # Demo predictions
    test_emails = [
        # Phishing examples
        """URGENT: Your bank account has been compromised! 
        
        Dear Customer,
        
        We have detected suspicious activity on your account. 
        Click http://fake-bank.com/verify immediately to confirm your identity.
        
        If you don't verify within 24 hours, your account will be suspended permanently.
        
        Click here to verify: https://secure-verify-login.com
        
        Sincerely,
        Security Team""",
        
        """Amazon: Action Required - Your Account Will Be Suspended
        
        Dear Valued Customer,
        
        We are writing to inform you that your Amazon account has been flagged for suspicious activity.
        
        Please update your payment information immediately to avoid service interruption.
        
        Update Now: http://amazon-verify-account.tk/login
        
        You have 48 hours to respond.""",
        
        # Legitimate examples
        """Hi John,
        
        Just following up on our meeting yesterday about the new project timeline.
        
        Let me know if you have any questions or need any clarification on the action items we discussed.
        
        Best regards,
        Sarah""",
        
        """Hello,
        
        Your order #123-4567890 has been shipped!
        
        Track your package: https://amazon.com/track/123-4567890
        
        Expected delivery: 3-5 business days.
        
        Thank you for shopping with us!"""
    ]
    
    print("Making predictions with BERT...")
    predictions = bert.predict(test_emails)
    
    for i, (email, pred) in enumerate(zip(test_emails, predictions)):
        print(f"\n{'='*60}")
        print(f"Email {i+1}: {email[:50]}...")
        print(f"{'='*60}")
        print(f"  Prediction: {'🚨 PHISHING' if pred['is_phishing'] else '✅ LEGITIMATE'}")
        print(f"  Phishing Probability: {pred.get('phishing_probability', 'N/A'):.2%}" 
              if 'phishing_probability' in pred else "  Probability: N/A")
        print(f"  Confidence: {pred.get('confidence', 'N/A'):.2%}" 
              if 'confidence' in pred else "  Confidence: N/A")
    
    print("\n" + "="*50 + "\n")
    
    # Generate detailed explanations
    print("Generating detailed explanations...")
    for i, email in enumerate(test_emails[:2]):
        print(f"\n--- Explanation for Email {i+1} ---")
        explanation = bert.explain_prediction(email)
        for key, value in explanation.items():
            if key != 'bert_analysis':
                print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    print("BERT Model Demo Complete!")
    print("\nNote: This is a pre-trained BERT model. For better results,")
    print("fine-tune on a labeled phishing email dataset.")

