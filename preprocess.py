"""
NLP Text Preprocessing Module for Phishing Detection

This module handles all NLP text cleaning and feature extraction tasks
for the phishing email detection system.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Tuple

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """
    NLP Text Preprocessor for phishing email detection.
    
    Handles text cleaning, tokenization, and feature extraction.
    """
    
    def __init__(self):
        """Initialize preprocessor with stopwords and urgency patterns."""
        self.stop_words = set(stopwords.words('english'))
        
        # Phishing urgency indicators
        self.urgency_words = {
            'urgent', 'immediate', 'immediately', 'action', 'required',
            'required', 'verify', 'verify', 'suspend', 'suspended',
            'terminate', 'terminated', 'close', 'closed', 'account',
            'bank', 'password', 'credential', 'security', 'alert',
            'warning', 'confirm', 'confirm', 'update', 'limited',
            'expires', '24', 'hours', '48', 'ASAP', 'deadline'
        }
        
        # Suspicious patterns
        self.suspicious_patterns = {
            'url_pattern': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'email_pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'money_pattern': r'\$[\d,]+(?:\.\d{2})?',
            'ip_pattern': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        }
    
    def clean_text(self, text: str) -> str:
        """
        NLP Cleaning Pipeline - Complete text preprocessing.
        
        Steps:
        1. Lowercase (NLP: Standardization)
        2. Replace URLs with [URL] token (NLP: Entity Recognition)
        3. Replace emails with [EMAIL] token
        4. Remove special characters but keep punctuation for analysis
        5. Remove extra whitespace
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Lowercase - Standardization
        text = text.lower()
        
        # 2. Replace URLs with [URL] token - Entity Recognition
        text = re.sub(
            self.suspicious_patterns['url_pattern'],
            '[URL]',
            text
        )
        
        # 3. Replace emails with [EMAIL] token
        text = re.sub(
            self.suspicious_patterns['email_pattern'],
            '[EMAIL]',
            text
        )
        
        # 4. Replace IP addresses with [IP] token
        text = re.sub(
            self.suspicious_patterns['ip_pattern'],
            '[IP]',
            text
        )
        
        # 5. Replace money amounts with [MONEY] token
        text = re.sub(
            self.suspicious_patterns['money_pattern'],
            '[MONEY]',
            text
        )
        
        # 6. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception:
            # Fallback to simple split if NLTK fails
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common stopwords from tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered tokens without stopwords
        """
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """
        NLP Feature Engineering - Extract features for ML models.
        
        Args:
            text: Raw email text
            
        Returns:
            Dictionary of extracted features
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        tokens_no_stop = self.remove_stopwords(tokens)
        
        features = {
            # Basic text statistics
            'char_count': len(text),
            'word_count': len(tokens),
            'avg_word_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            
            # Urgency indicators
            'urgency_word_count': self._count_urgency_words(tokens_no_stop),
            'has_urgency_words': self._has_urgency_words(tokens_no_stop),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            
            # Suspicious elements
            'url_count': len(re.findall(self.suspicious_patterns['url_pattern'], text)),
            'email_count': len(re.findall(self.suspicious_patterns['email_pattern'], text)),
            'ip_count': len(re.findall(self.suspicious_patterns['ip_pattern'], text)),
            'money_count': len(re.findall(self.suspicious_patterns['money_pattern'], text)),
            
            # Text analysis
            'uppercase_ratio': self._calculate_uppercase_ratio(text),
            'digit_ratio': self._calculate_digit_ratio(text),
            'special_char_ratio': self._calculate_special_char_ratio(text),
            
            # Token features
            'unique_word_ratio': len(set(tokens_no_stop)) / len(tokens_no_stop) if tokens_no_stop else 0,
            
            # Cleaned text for model input
            'cleaned_text': cleaned_text,
            'tokens': tokens_no_stop
        }
        
        return features
    
    def _count_urgency_words(self, tokens: List[str]) -> int:
        """Count urgency indicator words in tokens."""
        count = 0
        for token in tokens:
            if token.lower() in self.urgency_words:
                count += 1
        return count
    
    def _has_urgency_words(self, tokens: List[str]) -> int:
        """Check if any urgency words are present (binary)."""
        return 1 if self._count_urgency_words(tokens) > 0 else 0
    
    def _calculate_uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase characters."""
        if not text:
            return 0.0
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0
        return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    
    def _calculate_digit_ratio(self, text: str) -> float:
        """Calculate ratio of digit characters."""
        if not text:
            return 0.0
        return sum(1 for c in text if c.isdigit()) / len(text)
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters."""
        if not text:
            return 0.0
        special = set('!@#$%^&*()_+-=[]{}|;:,.<>?')
        return sum(1 for c in text if c in special) / len(text)
    
    def prepare_for_lstm(self, text: str, max_length: int = 100) -> List[int]:
        """
        Prepare text for LSTM model input.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of token indices
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        # Simple vocabulary (in production, use trained tokenizer)
        vocab = {
            '[PAD]': 0, '[UNK]': 1, '[URL]': 2, '[EMAIL]': 3,
            '[IP]': 4, '[MONEY]': 5
        }
        
        # Build vocabulary from training data in production
        # For now, use character-based encoding
        indices = []
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                # Use first char as placeholder
                indices.append(vocab['[UNK]'])
        
        # Pad or truncate
        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def prepare_for_bert(self, text: str) -> Dict[str, List[int]]:
        """
        Prepare text for BERT model input.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].tolist()[0],
            'attention_mask': encoded['attention_mask'].tolist()[0],
            'token_type_ids': encoded['token_type_ids'].tolist()[0]
        }


# Example usage and testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test with sample phishing email
    sample_email = """
    URGENT: Your account will be SUSPENDED within 24 hours!
    
    Dear Valued Customer,
    
    We have detected suspicious activity on your account. 
    Click http://fake-bank.com/verify immediately to confirm your identity.
    
    If you don't verify within 48 hours, your account will be TERMINATED.
    
    Verify now: https://secure-verify-12345.com/login
    
    Sincerely,
    Security Team
    """
    
    print("=== NLP Text Preprocessing Demo ===\n")
    print("Original Text:")
    print(sample_email)
    print("\n" + "="*50 + "\n")
    
    # Clean text
    cleaned = preprocessor.clean_text(sample_email)
    print("Cleaned Text:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Extract features
    features = preprocessor.extract_features(sample_email)
    print("Extracted Features:")
    for key, value in features.items():
        if key not in ['cleaned_text', 'tokens']:
            print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    print("Tokens (without stopwords):")
    print(features['tokens'])

