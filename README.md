# Dans la tête d'une IA

Une application web interactive permettant de visualiser et de comprendre le fonctionnement d'une IA en temps réel.

## Fonctionnalités Principales

- **Détection temps réel (YOLOv8)** : Streaming vidéo via **WebRTC** avec une latence quasi-nulle grâce à un système de frame-skipping dynamique.
- **Live Grad-CAM (GPU)** : Visualisation de l'attention de l'IA (Heatmaps) superposée en direct sur la webcam pour ouvrir la "boîte noire", calculé sur carte graphique.
- **Transfer Learning (ResNet)** : Entraînement interactif d'un modèle ResNet (18/50) sur le dataset CIFAR-10, exécuté dans un processus isolé (multiprocessing spawn) pour ne pas bloquer l'interface.

## Structure du projet

Ce projet est structuré en deux parties :
- **`frontend/`** : Interface utilisateur développée avec React et Vite (Hooks WebRTC, WebSockets, Composants dynamiques).
- **`backend/`** : Serveur asynchrone en Python (FastAPI, PyTorch, aiortc, OpenCV).

## Prérequis

Assurez-vous d'avoir les éléments suivants installés sur votre machine :
- **Node.js** (version 16+ recommandée)
- **Python** (version 3.8+)
- **Git**

---

## Installation et démarrage

### 1. Configuration du Backend (Python)

1. Ouvrez un terminal et déplacez-vous dans le dossier `backend` :
   ```bash
   cd backend
   ```
2. Créez un environnement virtuel (recommandé pour isoler les dépendances) :
   ```bash
   python3 -m venv venv
   ```
3. Activez l'environnement virtuel :
   - Sur **Linux/macOS** : `source venv/bin/activate`
   - Sur **Windows** : `venv\Scripts\activate`
4. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```
5. Lancez le serveur backend :
   ```bash
   python main.py
   ```

### 2. Configuration du Frontend (React/Vite)

1. Ouvrez un **nouveau** terminal et déplacez-vous dans le dossier `frontend` :
   ```bash
   cd frontend
   ```
2. Installez les dépendances Node.js :
   ```bash
   npm install
   ```
3. Lancez le serveur de développement Vite :
   ```bash
   npm run dev
   ```

Le frontend sera alors accessible depuis votre navigateur (généralement à l'adresse `http://localhost:5173`).

---

## Remarques importantes

- **Variables d'environnement** : Ne commitez jamais de fichiers `.env` contenant des clés API, des mots de passe ou d'autres données sensibles. Si des variables d'environnement sont nécessaires, créez un fichier `.env` en local dans les répertoires `frontend` ou `backend`. Ces fichiers sont ignorés par Git grâce au `.gitignore`.
- **Contribution** : Assurez-vous de bien tester les parties frontend et backend ensemble avant tout nouveau commit.
