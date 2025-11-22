# Plan d'Attaque : Robot Dessinateur Universel avec JEPA2

Ce document détaille la stratégie pour créer un robot capable de dessiner **tout type d'objet** en utilisant la base de données "Quick, Draw!" et une architecture de type World Model basée sur JEPA2.

## 1. Analyse des Ressources

### 1.1 La Base de Données "Quick, Draw!"
*   **Source** : [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
*   **Format Idéal** : `.ndjson` (Newline Delimited JSON).
*   **Contenu** :
    *   Chaque entrée contient une séquence de traits (strokes).
    *   Chaque trait est une liste de coordonnées `(x, y)` et de temps `t`.
    *   C'est un format **vectoriel**, parfait pour la robotique (contrairement aux pixels).
*   **Utilisation pour le projet** :
    *   Cette base servira à apprendre la "physique" du dessin humain et la représentation de **tous types d'objets**.
    *   On utilisera l'ensemble du dataset (345 catégories) pour apprendre une dynamique de trait généraliste et une compréhension sémantique large.

### 1.2 L'Architecture JEPA2 (V-JEPA / I-JEPA)
*   **Concept** : JEPA (Joint Embedding Predictive Architecture) apprend à prédire l'état futur d'un système dans un espace latent (abstrait), plutôt que de prédire les pixels exacts.
*   **Pourquoi JEPA2 pour ce projet ?**
    *   **Planification (MPC)** : Le robot doit planifier quel trait tracer pour se rapprocher de l'image finale. JEPA permet de simuler le résultat d'un trait dans l'espace latent.
    *   **Efficacité** : Prédire dans l'espace latent est beaucoup plus robuste et rapide que de générer des images pixel par pixel à chaque étape.

---

## 2. Évaluation : Modèle Pré-entraîné vs Entraînement "From Scratch"

Une question cruciale est de savoir si l'on peut utiliser un modèle **V-JEPA 2 pré-entraîné** (disponible via Meta/Hugging Face) ou si l'on doit tout réentraîner.

### Analyse de Viabilité du Pré-entraîné (V-JEPA 2)
*   **Nature du Pré-entraînement** : V-JEPA 2 est entraîné sur des **vidéos (pixels)** pour comprendre la dynamique du monde physique.
*   **Problème** : Nos données "Quick, Draw!" sont des **vecteurs**. Les poids du modèle pré-entraîné s'attendent à des patchs visuels (images), pas à des coordonnées (x,y).
*   **Solution Hybride (Recommandée)** :
    *   Utiliser V-JEPA 2 comme **Encodeur Visuel (Vision Encoder)**.
    *   On ne lui donne pas les vecteurs bruts, mais on **rend (dessine)** l'état actuel du canvas en image.
    *   **Avantage** : On profite de la puissance phénoménale de V-JEPA pour "comprendre" l'image (le dessin en cours + le portrait cible).
    *   **Inconvénient** : Nécessite une étape de rendu (rasterization) à chaque boucle, mais c'est acceptable.

### Architecture Révisée (Hybride)
1.  **Input** : Image du dessin en cours (Rendu ou Caméra) + Image cible.
2.  **Encoder (V-JEPA 2 Frozen/Fine-tuned)** : Extrait les features riches de ces images.
3.  **Predictor (À entraîner)** :
    *   Prend les features visuelles (de V-JEPA).
    *   Prend une **Action Vectorielle** (le trait à tracer).
    *   Predit le futur état latent.
    *   *C'est ce module léger qu'on entraîne avec Quick Draw.*

---

## 3. Architecture Proposée

L'idée centrale est d'utiliser JEPA comme un **World Model** (Modèle du Monde) qui comprend la dynamique du dessin.

### Composants Clés :

1.  **L'Encodeur (V-JEPA 2)** :
    *   Convertit l'état actuel de la feuille de dessin (canvas pixelisé) en une représentation latente $s_t$.
    *   Convertit également l'**Image Cible** (photo d'un objet, paysage, etc.) en une représentation latente cible $s_{target}$.

2.  **Le Prédicteur (World Model - Custom)** :
    *   Prend en entrée : L'état actuel latent $s_t$ et une **Action** potentielle $a$ (un trait de stylo : position, courbure, pression).
    *   Predit : Le futur état latent $s_{t+1}$.
    *   *C'est ici que le modèle apprend la "physique" du dessin à partir de Quick Draw.*

3.  **Le Planificateur (MPC - Model Predictive Control)** :
    *   C'est le "cerveau" qui décide quoi dessiner.
    *   À chaque étape, il génère plusieurs traits possibles.
    *   Il utilise le **Prédicteur** pour voir quel trait rapproche le plus l'état futur $s_{t+1}$ de l'état cible $s_{target}$ (dans l'espace latent).
    *   Il choisit la meilleure action et l'envoie au robot.

---

## 4. Plan d'Attaque (Étapes de Réalisation)

### Phase 1 : Préparation des Données ("Quick, Draw!")
1.  **Téléchargement** : Récupérer les fichiers `.ndjson` (commencer par une sélection variée de catégories, puis étendre à tout).
2.  **Parsing & Rendu** : Créer un script Python pour lire les vecteurs et générer des images "étapes par étapes" (ex: image après 1 trait, après 2 traits, etc.).
    *   *Objectif* : Créer des paires (Image État actuel, Action Vectorielle, Image État suivant) pour entraîner le Prédicteur.

### Phase 2 : Intégration V-JEPA & Entraînement
1.  **Setup V-JEPA** : Charger le modèle pré-entraîné (via Hugging Face) et figer ses poids (au début).
2.  **Entraînement du Prédicteur** :
    *   Entraîner uniquement la partie "Prédicteur" (petit réseau) qui fait le lien entre l'espace latent de V-JEPA et les actions vectorielles.
    *   Loss : Distance dans l'espace latent entre (Prediction) et (Encodage de l'image réelle suivante).

### Phase 3 : Le "Dessinateur" (Inférence & Planification)
1.  **Boucle de Contrôle** :
    *   Input : Une image quelconque (objet, animal, scène).
    *   Boucle :
        1.  Encoder le dessin actuel (vide au début) et la cible avec V-JEPA.
        2.  Générer X propositions de traits aléatoires.
        3.  Utiliser le Prédicteur pour évaluer quel trait minimise la distance avec la cible.
        4.  Valider le trait -> Mise à jour du dessin.
2.  **Optimisation** : Affiner le planificateur (Gradient Descent sur l'action au lieu de random sampling) pour des traits plus précis.

### Phase 4 : Interface Robotique (Optionnel / Futur)
1.  Convertir les "Actions" (vecteurs) en G-Code ou commandes moteurs pour le bras robotique.

---

## 5. Outils Recommandés
*   **Langage** : Python
*   **Framework DL** : PyTorch (implémentations JEPA existantes souvent en PyTorch).
*   **Modèles** : `transformers` (Hugging Face) pour V-JEPA 2.
*   **Données** : `ndjson` library, `numpy`, `cairo` ou `opencv` pour le rendu des vecteurs.

---

## 6. Implémentation Élémentaire du Monde du Dessin

Pour simplifier le problème et le rendre compatible avec une architecture World Model, nous définissons le "Monde" comme suit :

### 6.1 L'État (State) : La Matrice de Dessin
*   **Représentation** : Une matrice 2D (Image Raster) représentant la feuille de papier.
*   **Format** : Grayscale (1 canal), valeurs de 0 (blanc) à 1 (noir).
*   **Taille** : Fixe, par exemple `64x64` ou `128x128` pixels pour commencer.
*   **Pourquoi ?** C'est ce que "voit" l'encodeur V-JEPA. C'est la vérité terrain de l'état du monde.

### 6.2 L'Action : Le Trait Vectoriel
*   **Représentation** : Un vecteur représentant un segment de trait ou une courbe de Bézier simple.
*   **Format Élémentaire** : `(x_start, y_start, x_end, y_end, width, pressure)`
    *   Coordonnées normalisées entre 0 et 1.
    *   Alternative (si on garde en mémoire la position du stylo) : `(dx, dy, pen_down)`.
*   **Choix pour le projet** : Pour un World Model robuste, l'action **absolue** `(x1, y1, x2, y2)` est souvent plus simple à apprendre car elle ne dépend pas d'une variable cachée "position du stylo".

### 6.3 La Transition : Le Moteur de Rendu
*   **Fonction** : `NextState = Apply(CurrentState, Action)`
*   **Implémentation** : C'est un moteur de rendu déterministe (Rasterizer).
    *   On prend la matrice `CurrentState`.
    *   On dessine le trait défini par `Action` dessus (ex: avec `cv2.line` ou `PIL.ImageDraw`).
    *   On obtient `NextState`.
*   **Rôle dans le JEPA** :
    *   Le **Prédicteur** doit apprendre à *approximer* cette fonction de transition, mais dans l'**espace latent**.
    *   Au lieu de prédire les pixels de `NextState`, il prédit l'embedding de `NextState`.
