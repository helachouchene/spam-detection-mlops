@echo off
echo ========================================
echo 🚀 API Spam Detection - Version Complète
echo ========================================
echo.

echo 📦 Installation des dépendances...
pip install -r api/requirements.txt

echo.
echo 🔍 Vérification des modèles...
if not exist "models\logistic_regression_model.joblib" (
    echo ❌ Modèle non trouvé
    pause
    exit /b 1
)

echo ✅ Modèles OK
echo.
echo 🌐 Démarrage de l'API...
echo 📡 http://localhost:5000
echo.
echo 📝 Exemple d'utilisation:
echo curl -X POST http://localhost:5000/predict ^
echo      -H "Content-Type: application/json" ^
echo      -d "{\"message\": \"Congratulations! You won!\"}"
echo.
echo 🛑 Ctrl+C pour arrêter
echo ========================================
echo.

cd api
python app.py

pause
