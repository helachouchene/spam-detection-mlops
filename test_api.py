import requests
import json
import time

print("🧪 Test de l'API Spam Detection")
print("="*50)

# Attendre le démarrage
print("⏳ Attente du démarrage...")
time.sleep(3)

try:
    # Test 1: Health check
    print("\n1. 📍 Health check...")
    response = requests.get("http://localhost:5000/health", timeout=5)
    print(f"   Status: {response.status_code}")

    # Test 2: Prédiction SPAM
    print("\n2. 🔮 Prédiction SPAM...")
    data = {"message": "Congratulations! You won a free iPhone! Call now!"}
    response = requests.post("http://localhost:5000/predict", json=data, timeout=5)

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Prédiction: {result.get('prediction', 'N/A')}")
        print(f"   ✅ Probabilité SPAM: {result.get('spam_probability', 0):.2%}")
        print(f"   ✅ Temps: {result.get('processing_time_ms', 0)}ms")
    else:
        print(f"   ❌ Erreur: {response.json().get('error', 'Unknown')}")

    # Test 3: Prédiction HAM
    print("\n3. 🔮 Prédiction HAM...")
    data = {"message": "Hey, are we meeting tomorrow for lunch?"}
    response = requests.post("http://localhost:5000/predict", json=data, timeout=5)

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Prédiction: {result.get('prediction', 'N/A')}")
        print(f"   ✅ Probabilité SPAM: {result.get('spam_probability', 0):.2%}")

    # Test 4: Batch prediction
    print("\n4. 📦 Batch prediction...")
    data = {
        "messages": [
            "FREE entry to win £1000",
            "What time is the meeting?",
            "URGENT: Your account needs verification"
        ]
    }
    response = requests.post("http://localhost:5000/batch_predict", json=data, timeout=10)

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ {result.get('count', 0)} messages traités")
        for i, pred in enumerate(result.get('results', []), 1):
            print(f"   {i}. {pred.get('prediction')}: {pred.get('spam_probability', 0):.2%}")

except requests.exceptions.ConnectionError:
    print("\n❌ Impossible de se connecter à l'API")
    print("💡 Vérifie que l'API est démarrée: start_api.bat")
except Exception as e:
    print(f"\n❌ Erreur: {e}")

print("\n" + "="*50)
print("✅ Tests terminés")
