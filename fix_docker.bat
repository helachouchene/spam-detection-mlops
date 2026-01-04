@echo off
echo ========================================
echo 🐳 RÉPARATION DOCKER - SÉCURISÉ
echo ========================================
echo.

echo 🔍 Arrêt des services spam-detection seulement...
docker-compose down 2>nul

echo.
echo 🧹 Nettoyage spécifique à spam-detection...
docker images | findstr "spam-detection" && (
    echo Suppression des images spam-detection...
    docker rmi -f spam-detection:latest 2>nul
    docker rmi -f spam-detection-mlops-spam-api 2>nul
    docker rmi -f spam-detection-mlops-spam-ui 2>nul
)

echo.
echo 📦 Reconstruction de l'image...
docker build -t spam-detection:latest .

if %errorlevel% neq 0 (
    echo ❌ Erreur lors du build
    echo.
    echo 💡 Solutions:
    echo 1. Vérifie que Dockerfile n'a pas d'erreurs
    echo 2. Vérifie que requirements.txt existe
    echo 3. Essaye: docker build --no-cache -t spam-detection:latest .
    pause
    exit /b 1
)

echo ✅ Build réussi!
echo.

echo 🚀 Démarrage des services...
docker-compose up -d

echo.
echo ⏳ Attente du démarrage (5 secondes)...
timeout /t 5 /nobreak >nul

echo.
echo 🔍 Vérification des conteneurs spam-detection...
docker-compose ps

echo.
echo 🌐 Services disponibles:
echo    • 📡 API: http://localhost:5000
echo    • 🎨 Interface: http://localhost:8501
echo.
echo 📋 Commandes utiles:
echo    • Logs API: docker-compose logs api
echo    • Logs UI: docker-compose logs ui
echo    • Arrêter: docker-compose down
echo    • Redémarrer: docker-compose restart
echo.
echo 🧪 Pour tester: python test_api.py
echo.

pause