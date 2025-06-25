from flask import Flask, request, jsonify
import joblib
import langid
import numpy as np
import re
import string
from scipy.sparse import hstack
from Singlish2Sinhala import Translate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from datetime import datetime
import os

# Download NLTK data with fallback
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    try:
        # Fallback for older NLTK versions
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        NLTK_AVAILABLE = True
    except:
        print("Warning: NLTK download failed. Using basic text processing.")
        NLTK_AVAILABLE = False

app = Flask(__name__)

class ImprovedSpamAPI:
    def __init__(self):
        import os
        try:
            # Use os.path.join for cross-platform compatibility
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            
            self.english_model = joblib.load(os.path.join(models_dir, 'improved_english_spam_model.pkl'))
            self.sinhala_model = joblib.load(os.path.join(models_dir, 'improved_sinhala_spam_model.pkl'))
            self.english_vectorizer = joblib.load(os.path.join(models_dir, 'improved_english_vectorizer.pkl'))
            self.sinhala_vectorizer = joblib.load(os.path.join(models_dir, 'improved_sinhala_vectorizer.pkl'))
            self.feature_scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))
            
            try:
                self.english_nb_model = joblib.load(os.path.join(models_dir, 'english_nb_model.pkl'))
                self.sinhala_nb_model = joblib.load(os.path.join(models_dir, 'sinhala_nb_model.pkl'))
                self.has_nb_models = True
                print("Loaded separate NB models successfully")
            except:
                print("Warning: Separate NB models not found. Using ensemble only.")
                self.has_nb_models = False
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.english_model = None
            self.sinhala_model = None
            self.english_vectorizer = None
            self.sinhala_vectorizer = None
            self.feature_scaler = None
            self.has_nb_models = False
            print(f"Error loading models: {e}")
            raise
    
    def extract_features(self, text, language='en'):
        """Extract additional features from text (same as training)"""
        # Convert to string if not already
        text = str(text) if text is not None else ""
        
        features = {}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len([c for c in text if c.isalpha()])
        features['digit_count'] = len([c for c in text if c.isdigit()])
        features['upper_case_count'] = len([c for c in text if c.isupper()])
        features['punctuation_count'] = len([c for c in text if c in string.punctuation])
        
        # Ratios (avoid division by zero)
        if len(text) > 0:
            features['digit_ratio'] = features['digit_count'] / len(text)
            features['upper_ratio'] = features['upper_case_count'] / len(text)
            features['punct_ratio'] = features['punctuation_count'] / len(text)
        else:
            features['digit_ratio'] = 0
            features['upper_ratio'] = 0
            features['punct_ratio'] = 0
            
        # Special characters
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dollar_count'] = text.count('$')
        features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features['phone_count'] = len(re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b', text))
        
        # Enhanced Sinhala spam keywords
        spam_keywords_en = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'congratulations', 
                           'offer', 'deal', 'discount', 'limited', 'act now', 'call now', 'click here']
        
        # Expanded Sinhala spam keywords
        spam_keywords_si = [
            'නොමිලේ', 'ජයග්‍රහණය', 'ත්‍යාගය', 'මුදල්', 'හදිසි', 'සුභ පැතුම්',
            'ඉක්මන්', 'ජයග්‍රහීතා', 'නිර්ධන', 'ගෙවීම', 'ප්‍රිසාද', 'පරිසුදු',
            'ලොතරැයි', 'ක්ලික්', 'සම්බන්ධ', 'ලැබීම', 'ජය', 'ජයගත්', 'සුබ'
        ]
        
        if language == 'en':
            spam_words = spam_keywords_en
        else:
            spam_words = spam_keywords_si
            
        text_lower = text.lower()
        features['spam_keywords_count'] = sum(1 for word in spam_words if word in text_lower)
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0
            
        return list(features.values())
    
    def basic_tokenize(self, text):
        """Basic tokenization fallback when NLTK is not available"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def get_english_stopwords(self):
        """Get English stopwords with fallback"""
        if NLTK_AVAILABLE:
            try:
                return set(stopwords.words('english'))
            except:
                pass
        
        # Fallback stopwords list
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
    
    def preprocess_text(self, text, language='en'):
        """Enhanced text preprocessing with NLTK fallback"""
        # Convert to string if not already
        text = str(text) if text is not None else ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For English, remove stopwords but keep important ones
        if language == 'en':
            stop_words = self.get_english_stopwords()
            # Keep these words as they might be important for spam detection
            keep_words = {'free', 'win', 'money', 'cash', 'offer', 'deal', 'urgent', 'now'}
            stop_words = stop_words - keep_words
            
            if NLTK_AVAILABLE:
                try:
                    words = word_tokenize(text)
                except:
                    words = self.basic_tokenize(text)
            else:
                words = self.basic_tokenize(text)
                
            text = ' '.join([word for word in words if word not in stop_words])
        
        return text 
    
    def detect_language(self, text):
        """Detect language of the text"""
        try:
            language, confidence = langid.classify(text)
            return language, confidence
        except:
            return 'en', 0.5
    
    def rule_based_sinhala_spam_detection(self, text):
        """Rule-based Sinhala spam detection when model is inadequate"""
        text = str(text).lower()
        
        # Sinhala spam indicators
        spam_patterns = [
            r'නොමිලේ',  # freeP
            r'ජයග්‍රහණය',  # win
            r'ත්‍යාගය',  # prize
            r'මුදල්',  # money
            r'හදිසි',  # urgent
            r'ඉක්මන්',  # quick
            r'ලොතරැයි',  # lottery
            r'ජයගත්',  # won
            r'ක්ලික්',  # click
            r'රුපියල්.*\d+',  # rupees with numbers
            r'\d+.*රුපියල්',  # numbers with rupees
            r'අද.*අවසන්',  # today last/end
            r'සම්මාන.*ලබා',  # award/prize receive
            r'ලබා.*දිනුම',  # receive/win
            r'දුරකථන.*අංක',  # phone number
            r'ඔබ.*ජයගත්',  # you won
            r'සුභ.*පැතුම්',  # congratulations
        ]
        
        # Count spam indicators
        spam_score = 0
        for pattern in spam_patterns:
            if re.search(pattern, text):
                spam_score += 1
        
        # Check for excessive punctuation
        if text.count('!') > 2:
            spam_score += 1
        
        # Check for numbers (often used in prizes/money)
        if re.search(r'\d{3,}', text):  # 3 or more digits
            spam_score += 1
        
        # Check message length and excitement
        if len(text) > 100 and ('!' in text or '?' in text):
            spam_score += 0.5
        
        # Calculate probability based on spam score
        max_score = len(spam_patterns) + 3  # patterns + additional checks
        spam_probability = min(spam_score / max_score, 0.95)  # Cap at 95%
        ham_probability = 1 - spam_probability
        
        # Determine prediction
        prediction = 1 if spam_probability > 0.5 else 0
        confidence = max(spam_probability, ham_probability)
        
        return prediction, confidence, [ham_probability, spam_probability]
    
    def predict_spam(self, text):
        """Predict if text is spam with enhanced features"""
        try:
            # Convert to string if not already
            text = str(text) if text is not None else ""
            
            # Detect language
            detected_language, lang_confidence = self.detect_language(text)
            
            print(f"Detected language: {detected_language} (confidence: {lang_confidence:.2f})")
            
            if detected_language == 'en':
                # English processing
                processed_text = self.preprocess_text(text, 'en')
                text_features = self.english_vectorizer.transform([processed_text])
                additional_features = np.array([self.extract_features(text, 'en')])
                additional_features_scaled = self.feature_scaler.transform(additional_features)
                
                # Combine features
                combined_features = hstack([text_features, additional_features_scaled])
                
                # Make predictions
                if self.has_nb_models:
                    # Get predictions from both models
                    ensemble_pred_proba = self.english_model.predict_proba(combined_features)[0]
                    nb_pred_proba = self.english_nb_model.predict_proba(text_features)[0]
                    
                    # Combine predictions
                    combined_proba = (ensemble_pred_proba + nb_pred_proba) / 2
                    prediction = (combined_proba[1] > 0.5).astype(int)
                    confidence_scores = combined_proba
                else:
                    # Use only ensemble model
                    prediction = self.english_model.predict(combined_features)[0]
                    confidence_scores = self.english_model.predict_proba(combined_features)[0]
                
                confidence = max(confidence_scores)
                
                result = {
                    'prediction': 'Spam' if prediction == 1 else 'Ham',
                    'confidence': float(confidence),
                    'language': 'English',
                    'language_confidence': float(lang_confidence),
                    'prediction_probabilities': {
                        'ham': float(confidence_scores[0]),
                        'spam': float(confidence_scores[1])
                    }
                }
                
            elif detected_language == 'si':
                # Sinhala processing - Use rule-based approach if model is inadequate
                if hasattr(self.sinhala_model, 'strategy'):
                    # Use rule-based detection for better results
                    print("Using rule-based Sinhala spam detection")
                    prediction, confidence, confidence_scores = self.rule_based_sinhala_spam_detection(text)
                    
                    result = {
                        'prediction': 'Spam' if prediction == 1 else 'Ham',
                        'confidence': float(confidence),
                        'language': 'Sinhala',
                        'language_confidence': float(lang_confidence),
                        'prediction_probabilities': {
                            'ham': float(confidence_scores[0]),
                            'spam': float(confidence_scores[1])
                        },
                        'detection_method': 'rule-based'
                    }
                else:
                    # Use trained model
                    text_features = self.sinhala_vectorizer.transform([text])
                    additional_features = np.array([self.extract_features(text, 'si')])
                    additional_features_scaled = self.feature_scaler.transform(additional_features)
                    
                    # Combine features
                    combined_features = hstack([text_features, additional_features_scaled])
                    
                    # Make predictions
                    if self.has_nb_models and hasattr(self, 'sinhala_nb_model'):
                        # Get predictions from both models
                        ensemble_pred_proba = self.sinhala_model.predict_proba(combined_features)[0]
                        nb_pred_proba = self.sinhala_nb_model.predict_proba(text_features)[0]
                        
                        # Combine predictions
                        combined_proba = (ensemble_pred_proba + nb_pred_proba) / 2
                        prediction = (combined_proba[1] > 0.5).astype(int)
                        confidence_scores = combined_proba
                    else:
                        # Use only ensemble model
                        prediction = self.sinhala_model.predict(combined_features)[0]
                        confidence_scores = self.sinhala_model.predict_proba(combined_features)[0]
                    
                    confidence = max(confidence_scores)
                    
                    result = {
                        'prediction': 'Spam' if prediction == 1 else 'Ham',
                        'confidence': float(confidence),
                        'language': 'Sinhala',
                        'language_confidence': float(lang_confidence),
                        'prediction_probabilities': {
                            'ham': float(confidence_scores[0]),
                            'spam': float(confidence_scores[1])
                        },
                        'detection_method': 'model-based'
                    }
                
            else:
                # Translate to Sinhala and process
                try:
                    translated_text = Translate(text)
                    print(f"Translated text: {translated_text}")
                    
                    # Use rule-based detection on translated text
                    prediction, confidence, confidence_scores = self.rule_based_sinhala_spam_detection(translated_text)
                    
                    result = {
                        'prediction': 'Spam' if prediction == 1 else 'Ham',
                        'confidence': float(confidence),
                        'language': f'Other ({detected_language}) - Translated to Sinhala',
                        'language_confidence': float(lang_confidence),
                        'translated_text': translated_text,
                        'prediction_probabilities': {
                            'ham': float(confidence_scores[0]),
                            'spam': float(confidence_scores[1])
                        },
                        'detection_method': 'rule-based-translated'
                    }
                    
                except Exception as translate_error:
                    print(f"Translation error: {translate_error}")
                    # Fallback to English processing
                    return self.predict_spam_as_english(text, detected_language, lang_confidence)
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'error': str(e),
                'prediction': 'Unknown',
                'confidence': 0.0
            }
    
    def predict_spam_as_english(self, text, detected_language, lang_confidence):
        """Fallback: process as English when translation fails"""
        try:
            processed_text = self.preprocess_text(text, 'en')
            text_features = self.english_vectorizer.transform([processed_text])
            additional_features = np.array([self.extract_features(text, 'en')])
            additional_features_scaled = self.feature_scaler.transform(additional_features)
            
            combined_features = hstack([text_features, additional_features_scaled])
            
            if self.has_nb_models:
                ensemble_pred_proba = self.english_model.predict_proba(combined_features)[0]
                nb_pred_proba = self.english_nb_model.predict_proba(text_features)[0]
                combined_proba = (ensemble_pred_proba + nb_pred_proba) / 2
                prediction = (combined_proba[1] > 0.5).astype(int)
                confidence_scores = combined_proba
            else:
                prediction = self.english_model.predict(combined_features)[0]
                confidence_scores = self.english_model.predict_proba(combined_features)[0]
            
            confidence = max(confidence_scores)
            
            return {
                'prediction': 'Spam' if prediction == 1 else 'Ham',
                'confidence': float(confidence),
                'language': f'Other ({detected_language}) - Processed as English (fallback)',
                'language_confidence': float(lang_confidence),
                'prediction_probabilities': {
                    'ham': float(confidence_scores[0]),
                    'spam': float(confidence_scores[1])
                }
            }
        except Exception as e:
            return {
                'error': f'Fallback processing failed: {str(e)}',
                'prediction': 'Unknown',
                'confidence': 0.0
            }

# Initialize the improved spam detector
spam_detector = ImprovedSpamAPI()

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text_content = data['text']
        client_ip = request.remote_addr
        timestamp = datetime.now().isoformat()

        result = spam_detector.predict_spam(text_content)

        # Build log entry
        log_entry = {
            "timestamp": timestamp,
            "client_ip": client_ip,
            "request": data,
            "response": result
        }

        # Save to logs/requests_log.json (append mode)
        log_dir = '/app/logs'  # Docker path
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'requests_log.json')

        with open(log_path, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')  # One JSON object per line

        return jsonify(result)

    except Exception as e:
        logging.error(f" Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Improved spam detection API is running'})

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded models"""
    return jsonify({
        'english_model': str(type(spam_detector.english_model)),
        'sinhala_model': str(type(spam_detector.sinhala_model)),
        'has_nb_models': spam_detector.has_nb_models,
        'features': 'Enhanced with additional text features, ensemble models, and rule-based Sinhala detection',
        'sinhala_detection': 'Rule-based pattern matching for better accuracy'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)