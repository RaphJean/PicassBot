# Journal de Bord - Picassbot

Ce document résume les étapes clés et les réalisations du projet depuis son lancement.

## 1. Initialisation et Infrastructure
-   **Setup** : Initialisation du projet avec `uv`, configuration de l'environnement Python.
-   **Moteur de Dessin (`DrawingWorld`)** :
    -   Création d'un environnement de dessin basé sur `numpy` et `PIL`.
    -   Passage d'actions absolues à des **actions relatives** (`dx`, `dy`) pour plus de flexibilité.
    -   Support des actions discrètes : `eos` (end of stroke) et `eod` (end of drawing).
    -   Rendu vectoriel simulé sur un canvas pixelisé (128x128).

## 2. Données (QuickDraw)
-   **Téléchargement** : Script pour télécharger l'intégralité du dataset QuickDraw (format `.ndjson` converti en `.npz` ou utilisé brut).
-   **Parsing** : Implémentation de `QuickDrawDataset` pour charger, parser et rendre les dessins vectoriels.
-   **Format** : Gestion du format spécifique de QuickDraw (strokes séparés, coordonnées relatives).

## 3. Stratégies de Recherche (Planning)
Implémentation de plusieurs algorithmes pour trouver la séquence d'actions optimale pour reproduire une image cible :
-   **Greedy Search** : Optimisation locale pas à pas. Rapide mais myope.
-   **MPC (Random Shooting)** : Planification sur un horizon court (ex: 5 steps) en tirant des séquences aléatoires.
-   **Genetic Algorithm** : Évolution d'une population de séquences d'actions.
-   **CEM (Cross-Entropy Method)** : Optimisation itérative de la distribution des actions (Gaussienne).
-   **MCTS (Monte Carlo Tree Search)** :
    -   Exploration d'arbre avec UCB (Upper Confidence Bound).
    -   Gestion des statistiques de visite et de coût.
    -   **Amélioration** : Ajout d'une option "Early Stopping" globale.

## 4. Policy Network (Apprentissage)
Création d'un "cerveau" pour guider le dessin, entraîné par Imitation Learning sur QuickDraw.
-   **Architecture** :
    -   **Dual Encoder** : Deux CNNs (ResNet-like) pour encoder l'état courant du canvas et l'image cible.
    -   **Fusion** : Concaténation des embeddings.
    -   **Action Heads** :
        -   Continue : Prédiction de la moyenne et log-std pour `dx`, `dy` (Distribution Gaussienne).
        -   Discrète : Logits pour `eos` et `eod`.
-   **Entraînement** :
    -   Boucle d'entraînement PyTorch.
    -   Support de l'accélération matérielle **MPS (Apple Silicon)** pour M2.
    -   Loss : NLL (Negative Log Likelihood) pour le continu + BCE pour le discret.

## 5. Intégration Policy-Guided Planning
Fusion des deux mondes (Planning + Learning).
-   **PolicyStrategy** : Utilisation directe du réseau pour dessiner (inférence pure).
-   **Guidage** : Modification de `MCTS` et `MPC` pour échantillonner les actions depuis la distribution prédite par la Policy, au lieu d'un bruit aléatoire uniforme.
-   **Résultat** : Recherche beaucoup plus efficace et "humaine".

## 6. Expérimentation
-   **CLI (`run_experiments.py`)** : Interface ligne de commande complète pour lancer n'importe quelle stratégie.
-   **Cibles Flexibles** :
    -   Formes géométriques (Carré, Rond, Triangle).
    -   Images réelles du dataset (ex: "cat") pour tester la généralisation.
-   **Visualisation** : Génération automatique de GIFs montrant le processus de dessin étape par étape.

---
**État Actuel** : Le système est capable de dessiner en utilisant soit des algorithmes de recherche pure, soit une policy apprise, soit une combinaison des deux (MCTS guidé par Policy).

## 7. Joint JEPA Model & Latent MPC
Implémentation d'une architecture inspirée de JEPA (Joint Embedding Predictive Architecture) pour apprendre un modèle du monde latent.
-   **Architecture (`FullAgent`)** :
    -   **Encoder Partagé** : Encode le canvas et la target.
    -   **Latent Predictor** : Prédit l'état latent futur $z_{t+1}$ à partir de $z_t$ et de l'action $a_t$.
    -   **Policy Head** : Prédit l'action à partir de $z_t$ et $z_{target}$.
-   **Latent MPC** :
    -   Planification entièrement dans l'espace latent (plus rapide et sémantique).
    -   Optimisation de la séquence d'actions pour minimiser la distance $\|z_{pred} - z_{target}\|^2$.
-   **Améliorations de la Stabilité** :
    -   **Fix NaN Loss** : Clamping de `logstd` ([-20, 2]), Gradient Clipping, et Epsilon pour la variance.
    -   **Fix Latent Collapse** : Ajout d'une **Variance Regularization** pour forcer l'encodeur à ne pas produire des vecteurs constants (ce qui rendait la loss du prédicteur nulle mais inutile).
    -   **Monitoring** : Logging de la variance des latents (`z_std`) et évaluation visuelle (dessin d'un carré) à chaque époque.
-   **Infrastructure** :
    -   Ajout du support **Google Colab** (`colab_training.ipynb`) pour l'entraînement sur GPU.
    -   Possibilité d'initialiser l'encodeur joint à partir d'une Policy pré-entraînée.
