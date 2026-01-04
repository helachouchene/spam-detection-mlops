from flask import Flask, request, jsonify
import joblib
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import time
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import warnings
import os  # <-- AJOUT IMPORT OBLIGATOIRE POUR CLOUD
warnings.filterwarnings('ignore')

# ============================================
# INITIALISATION
# ============================================

app = Flask(__name__)

print("🚀 Chargement des modèles Spam Detection...")

# Charger les modèles
try:
    # CHEMIN RELATIF CORRIGÉ POUR CLOUD
    model = joblib.load('models/logistic_regression_model_final.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer_final.joblib')
    
    # Charger le label_encoder (peut être un dict ou un LabelEncoder)
    label_encoder = joblib.load('models/label_encoder_final.joblib')
    
    # Vérifier le type de label_encoder
    if isinstance(label_encoder, dict):
        print("ℹ️  LabelEncoder détecté comme dictionnaire")
        # Créer une fonction de décodage pour le dict
        def decode_label(pred):
            return 'spam' if pred == 1 else 'ham'
        label_decoder = decode_label
    else:
        # C'est un vrai LabelEncoder sklearn
        print("ℹ️  LabelEncoder détecté comme objet sklearn")
        def decode_label(pred):
            return label_encoder.inverse_transform([pred])[0]
        label_decoder = decode_label
    
except Exception as e:
    print(f"❌ Erreur de chargement: {e}")
    # En production, on ne lève pas d'exception
    model = None
    vectorizer = None
    label_decoder = lambda pred: 'ham'

print(f"✅ Modèles chargés")
print(f"   • Modèle: {type(model).__name__ if model else 'None'}")
print(f"   • Features attendues: {model.n_features_in_ if model else 'N/A'}")

# NLTK
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')

# ============================================
# FONCTIONS DE PRÉTRAITEMENT
# ============================================

def clean_text(text):
    """Nettoie le texte"""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    sms_stop_words = {'u', 'ur', 'im', 'gt', 'lt', 'amp', 'll', 've', 'dont', 'cant', 'wont'}
    stop_words.update(sms_stop_words)
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def extract_numeric_features(text):
    """Extrait les 16 features numériques"""
    # Longueur
    char_count = len(text)
    word_count = len(text.split())
    avg_word_length = char_count / max(word_count, 1)

    # Mots suspects
    spam_keywords = ['free', 'win', 'cash', 'prize', 'claim', 'urgent', 'offer', 'congratulations']
    keyword_features = []
    for keyword in spam_keywords:
        keyword_features.append(1 if keyword in text.lower() else 0)

    # Ponctuation
    exclamation_count = text.count('!')
    question_count = text.count('?')
    upper_case_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Flags
    is_long_message = 1 if char_count > 100 else 0
    has_punctuation = 1 if ('!' in text or '?' in text) else 0

    # Compiler
    features = [
        char_count,
        word_count,
        avg_word_length,
        *keyword_features,
        exclamation_count,
        question_count,
        upper_case_ratio,
        is_long_message,
        has_punctuation
    ]

    return np.array(features, dtype=np.float32)

def prepare_features(message):
    """Prépare toutes les features pour la prédiction"""
    if not model or not vectorizer:
        return None
    
    try:
        # Nettoyer
        cleaned = clean_text(message)

        # TF-IDF
        text_features = vectorizer.transform([cleaned])

        # Numériques
        numeric_features = extract_numeric_features(message)
        numeric_features_sparse = csr_matrix(numeric_features.reshape(1, -1))

        # Combiner
        all_features = hstack([text_features, numeric_features_sparse])

        return all_features
    except Exception:
        return None

# ============================================
# ENDPOINTS API
# ============================================

@app.route('/')
def home():
    """Page d'accueil"""
    return jsonify({
        'api': 'Spam Detection API',
        'version': '1.0.0',
        'status': 'running' if model else 'error',
        'model': type(model).__name__ if model else 'None',
        'features': model.n_features_in_ if model else 0
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model': 'LogisticRegression' if model else 'None',
        'features_ok': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction d'un message"""
    try:
        if not model or not vectorizer:
            return jsonify({
                'success': False,
                'error': 'Modèle non chargé',
                'timestamp': datetime.now().isoformat()
            }), 500

        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'Le champ "message" est requis'}), 400

        message = data['message']
        threshold = data.get('threshold', 0.5)

        start_time = time.time()

        # Préparer les features
        features = prepare_features(message)
        
        if features is None:
            return jsonify({
                'success': False,
                'error': 'Erreur préparation features',
                'timestamp': datetime.now().isoformat()
            }), 500

        # Prédire
        prediction = model.predict(features)[0]
        
        # Probabilités
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            spam_prob = probabilities[1]
            ham_prob = probabilities[0]
        else:
            spam_prob = 1.0 if prediction == 1 else 0.0
            ham_prob = 1.0 - spam_prob

        # Appliquer seuil
        if threshold != 0.5:
            prediction = 1 if spam_prob >= threshold else 0

        # Décoder avec la bonne méthode
        label = label_decoder(prediction)

        # Réponse
        return jsonify({
            'success': True,
            'message': message[:200] + "..." if len(message) > 200 else message,
            'prediction': label,
            'spam_probability': float(spam_prob),
            'ham_probability': float(ham_prob),
            'threshold': float(threshold),
            'confidence': 'HIGH' if max(spam_prob, ham_prob) > 0.8 else 'MEDIUM' if max(spam_prob, ham_prob) > 0.6 else 'LOW',
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'features_used': {
                'tfidf': vectorizer.transform([clean_text(message)]).shape[1] if vectorizer else 0,
                'numeric': 16,
                'total': features.shape[1] if features is not None else 0
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Prédiction batch"""
    try:
        if not model or not vectorizer:
            return jsonify({
                'success': False,
                'error': 'Modèle non chargé'
            }), 500

        data = request.get_json()

        if not data or 'messages' not in data:
            return jsonify({'error': 'Le champ "messages" est requis'}), 400

        messages = data['messages']
        threshold = data.get('threshold', 0.5)

        if not isinstance(messages, list):
            return jsonify({'error': '"messages" doit être une liste'}), 400

        results = []
        for msg in messages[:20]:  # Limiter à 20 messages
            features = prepare_features(str(msg))
            
            if features is None:
                continue
                
            prediction = model.predict(features)[0]
            
            # Probabilités
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                spam_prob = probabilities[1]
            else:
                spam_prob = 1.0 if prediction == 1 else 0.0

            # Appliquer seuil
            if threshold != 0.5:
                prediction = 1 if spam_prob >= threshold else 0

            label = label_decoder(prediction)

            results.append({
                'message': str(msg)[:100],
                'prediction': label,
                'spam_probability': float(spam_prob),
                'confidence': 'HIGH' if spam_prob > 0.8 or spam_prob < 0.2 else 'MEDIUM' if spam_prob > 0.6 or spam_prob < 0.4 else 'LOW'
            })

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# DÉMARRAGE POUR CLOUD
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🌐 API Spam Detection démarrée sur le port {port}")
    print(f"📡 http://0.0.0.0:{port}")
    print("\n📋 Endpoints disponibles:")
    print("   • GET  /          - Documentation")
    print("   • GET  /health    - Health check")
    print("   • POST /predict   - Prédire un message")
    print("   • POST /batch_predict - Prédire plusieurs messages")
    print("\n🚀 Serveur démarré en mode production!")
    app.run(host='0.0.0.0', port=port, debug=False)