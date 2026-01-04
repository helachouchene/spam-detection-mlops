# Dockerfile corrigé pour Cloud Run
FROM python:3.9-slim

WORKDIR /app

# Copier requirements
COPY requirements.txt requirements_streamlit.txt ./

# Installer dépendances + gunicorn
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements_streamlit.txt \
    gunicorn

# NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copier application
COPY . .

# Créer fichier __init__.py pour l'API
RUN echo "from .app import app" > api/__init__.py && \
    echo "" >> api/__init__.py && \
    echo "__all__ = ['app']" >> api/__init__.py

# Variable d'environnement pour le port (Cloud Run utilise PORT)
ENV PORT=8080

# Commande de démarrage avec gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 api:app