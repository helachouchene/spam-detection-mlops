# app_streamlit.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .spam-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .ham-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<h1 class="main-header">🛡️ Spam Detection Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3095/3095583.png", width=100)
    st.title("Configuration")
    
    api_url = st.text_input("API URL", "http://localhost:5000")
    
    threshold = st.slider(
        "Seuil de classification",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probabilité minimale pour classer comme spam"
    )
    
    st.markdown("---")
    st.info("""
    **Comment utiliser:**
    1. Entrez un message à analyser
    2. Ajustez le seuil si nécessaire
    3. Visualisez les résultats
    """)

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyse", "📊 Batch", "📈 Stats", "ℹ️ Info"])

with tab1:
    # Section analyse unique
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Analyse de message")
        message = st.text_area(
            "Entrez votre message:",
            height=150,
            placeholder="Ex: Congratulations! You won a free iPhone..."
        )
        
        if st.button("🔎 Analyser", type="primary", use_container_width=True):
            if message.strip():
                with st.spinner("Analyse en cours..."):
                    try:
                        # Appel API
                        response = requests.post(
                            f"{api_url}/predict",
                            json={
                                "message": message,
                                "threshold": threshold
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Affichage résultat
                            if result['success']:
                                pred_class = result['prediction']
                                spam_prob = result['spam_probability']
                                
                                # Box colorée
                                box_class = "spam-box" if pred_class == "spam" else "ham-box"
                                st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
                                
                                col_a, col_b, col_c = st.columns(3)
                                
                                with col_a:
                                    st.metric(
                                        "Prédiction",
                                        pred_class.upper(),
                                        delta="SPAM" if pred_class == "spam" else "HAM"
                                    )
                                
                                with col_b:
                                    st.metric(
                                        "Probabilité SPAM",
                                        f"{spam_prob:.2%}",
                                        delta=f"Seuil: {threshold}"
                                    )
                                
                                with col_c:
                                    confidence = result.get('confidence', 'MEDIUM')
                                    st.metric("Confiance", confidence)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Graphique jauge
                                fig, ax = plt.subplots(figsize=(8, 2))
                                ax.barh(['SPAM'], [spam_prob], color='#FF6B6B' if pred_class == 'spam' else '#4ECDC4')
                                ax.barh(['SPAM'], [1-spam_prob], left=[spam_prob], color='#C7F0DB')
                                ax.set_xlim(0, 1)
                                ax.axvline(x=threshold, color='red', linestyle='--', label=f'Seuil ({threshold})')
                                ax.set_xlabel('Probabilité')
                                ax.legend()
                                st.pyplot(fig)
                                
                                # Détails techniques
                                with st.expander("🔧 Détails techniques"):
                                    st.json(result)
                                    
                            else:
                                st.error(f"Erreur: {result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"Erreur API: {response.status_code}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Impossible de se connecter à l'API")
                        st.info("Assurez-vous que l'API est démarrée: `python api/app.py`")
            else:
                st.warning("Veuillez entrer un message à analyser")
    
    with col2:
        st.subheader("📋 Exemples")
        examples = {
            "SPAM Evident": "WINNER! You won 1 million dollars! Click here to claim!",
            "SPAM Subtile": "Your account needs verification. Please confirm your details.",
            "HAM Normal": "Hey, are we still meeting for lunch tomorrow?",
            "HAM Professionnel": "Meeting rescheduled to 3 PM. Please confirm attendance."
        }
        
        for title, example in examples.items():
            if st.button(title, use_container_width=True):
                st.session_state.message = example
                st.rerun()

with tab2:
    # Analyse batch
    st.subheader("📦 Analyse Batch")
    
    uploaded_file = st.file_uploader(
        "Téléchargez un fichier CSV/TXT",
        type=['csv', 'txt'],
        help="Fichier avec un message par ligne"
    )
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if 'message' not in df.columns:
                st.error("Le CSV doit contenir une colonne 'message'")
            else:
                messages = df['message'].tolist()
        else:
            messages = uploaded_file.getvalue().decode('utf-8').splitlines()
        
        st.write(f"📄 {len(messages)} messages trouvés")
        
        if st.button("🚀 Analyser tous les messages", type="primary"):
            with st.spinner(f"Analyse de {len(messages)} messages..."):
                try:
                    response = requests.post(
                        f"{api_url}/batch_predict",
                        json={
                            "messages": messages[:50],  # Limite à 50
                            "threshold": threshold
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        if results['success']:
                            # Créer DataFrame résultats
                            df_results = pd.DataFrame(results['results'])
                            
                            # Afficher tableau
                            st.dataframe(
                                df_results.style.apply(
                                    lambda x: ['background-color: #FFEBEE' if v == 'spam' 
                                              else 'background-color: #E8F5E9' 
                                              for v in x],
                                    subset=['prediction']
                                ),
                                use_container_width=True
                            )
                            
                            # Statistiques
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total", results['count'])
                            
                            with col2:
                                spam_count = sum(1 for r in results['results'] if r['prediction'] == 'spam')
                                st.metric("SPAM détectés", spam_count)
                            
                            with col3:
                                avg_prob = sum(r['spam_probability'] for r in results['results']) / len(results['results'])
                                st.metric("Probabilité moyenne", f"{avg_prob:.2%}")
                            
                            with col4:
                                high_conf = sum(1 for r in results['results'] if r['confidence'] == 'HIGH')
                                st.metric("Haute confiance", high_conf)
                            
                            # Télécharger résultats
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger résultats CSV",
                                data=csv,
                                file_name=f"spam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                    else:
                        st.error(f"Erreur API: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

with tab3:
    # Statistiques
    st.subheader("📈 Statistiques et Visualisations")
    
    try:
        # Charger modèle pour stats
        model = joblib.load('models/logistic_regression_model_final.joblib')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Features totales", model.n_features_in_)
            st.caption("TF-IDF + Numériques")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Graphique features importance (top 20)
            if hasattr(model, 'coef_'):
                st.subheader("Top 20 Features Importantes")
                coefs = model.coef_[0]
                
                # Séparer TF-IDF et numériques
                tfidf_coefs = coefs[:1000]
                numeric_coefs = coefs[1000:]
                
                # Top TF-IDF features
                vectorizer = joblib.load('models/tfidf_vectorizer_final.joblib')
                feature_names = list(vectorizer.get_feature_names_out()) + [
                    'char_count', 'word_count', 'avg_word_len',
                    'free', 'win', 'cash', 'prize', 'claim', 
                    'urgent', 'offer', 'congrats',
                    'excl_count', 'quest_count', 'upper_ratio',
                    'is_long', 'has_punct'
                ]
                
                top_indices = tfidf_coefs.argsort()[-20:][::-1]
                top_features = pd.DataFrame({
                    'Feature': [feature_names[i] for i in top_indices],
                    'Importance': tfidf_coefs[top_indices]
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_features, y='Feature', x='Importance', ax=ax)
                ax.set_title('Top 20 Features pour Détection SPAM')
                st.pyplot(fig)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Accuracy Modèle", "97.8%")
            st.caption("Sur données de test")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Distribution des prédictions batch
            st.subheader("Distribution des Prédictions")
            
            # Données simulées pour le graphique
            labels = ['SPAM', 'HAM']
            sizes = [15, 85]  # Approximation basée sur dataset
            
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                  colors=['#FF6B6B', '#4ECDC4'], startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            
    except Exception as e:
        st.warning(f"Statistiques limitées: {str(e)}")

with tab4:
    # Informations projet
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=200)
    
    with col2:
        st.header("ℹ️ À propos du Projet")
        
        st.markdown("""
        ### 📋 Description
        Système de détection de spam utilisant le Machine Learning avec pipeline MLOps complet.
        
        ### 🚀 Fonctionnalités
        - **Analyse en temps réel** des messages
        - **Batch processing** pour fichiers multiples
        - **Interface intuitive** avec visualisations
        - **API REST** pour intégration
        
        ### 🛠️ Technologie
        - **Backend**: Python, Flask, Scikit-learn
        - **ML**: Logistic Regression (97.8% accuracy)
        - **Frontend**: Streamlit
        - **Déploiement**: Docker
        
        ### 📊 Données
        - **Dataset**: SMS Spam Collection (5,572 messages)
        - **Classes**: SPAM (13.4%) / HAM (86.6%)
        - **Features**: 1016 (TF-IDF + numériques)
        """)
    
    st.markdown("---")
    
    # Status API
    st.subheader("🔧 Status du Système")
    
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if health_response.status_code == 200:
                st.success("✅ API Opérationnelle")
            else:
                st.error("❌ API Hors ligne")
        
        with col2:
            st.metric("Response Time", f"{health_response.elapsed.total_seconds()*1000:.0f} ms")
        
        with col3:
            st.metric("Version", "1.0.0")
            
    except:
        st.error("❌ API Non accessible")
        st.info("Démarrez l'API: `python api/app.py`")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🛡️ Spam Detection MLOps • Projet Complet • 
        <a href='https://github.com/helachouchene/spam-detection-mlops' target='_blank'>GitHub</a> •
        Développé avec ❤️
    </div>
    """,
    unsafe_allow_html=True
)