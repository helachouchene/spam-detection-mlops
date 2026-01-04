@echo off
echo 🐳 Construction de l'image Docker pour Spam Detection...
echo.

echo 📦 Building Docker image...
docker build -t spam-detection-mlops:latest .

echo.
echo 🚀 Démarrage des services...
docker-compose up -d

echo.
echo ✅ Services démarrés avec succès!
echo.
echo 🌐 Accès aux services:
echo    • 📡 API Flask:    http://localhost:5000
echo    • 🎨 Interface:    http://localhost:8501
echo.
echo 📋 Commandes utiles:
echo    • Voir les logs:    docker-compose logs -f
echo    • Arrêter:          docker-compose down
echo    • Redémarrer:       docker-compose restart
echo    • Status:           docker-compose ps
echo.

timeout /t 3 /nobreak >nul

echo 🔍 Vérification des services...
docker-compose ps

echo.
echo 🎉 Prêt! Ouvrez votre navigateur sur les URLs ci-dessus.
pause