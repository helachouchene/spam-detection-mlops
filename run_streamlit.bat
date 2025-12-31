@echo off
echo 🎨 Démarrage de l'interface Streamlit...
echo.

echo 📦 Installation des dépendances...
pip install -r requirements_streamlit.txt

echo.
echo 🌐 Démarrage de l'interface...
echo 📊 http://localhost:8501
echo.

streamlit run app_streamlit.py

pause