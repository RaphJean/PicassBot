# Méthodologie JEPA (Joint Embedding Predictive Architecture)

Ce document détaille l'approche **Self-Supervised Learning (SSL)** implémentée pour apprendre un modèle du monde robuste pour Picassbot.

## 1. Concept Fondamental
L'objectif est d'apprendre une représentation latente $z$ de l'environnement qui capture la sémantique et la dynamique, sans utiliser de labels (comme les classes "chat" ou "chien") et sans se focaliser sur la reconstruction de pixels (comme un Auto-Encoder).

Nous utilisons une architecture **JEPA** inspirée de *I-JEPA* et *BYOL*.

## 2. Architecture

Le modèle (`JEPAWorldModel`) est composé de trois parties :

### A. Online Encoder ($f_{\theta}$)
-   **Rôle** : Encoder l'état courant $s_t$.
-   **Mise à jour** : Descente de gradient standard.
-   **Sortie** : $z_t = f_{\theta}(s_t)$

### B. Target Encoder ($f_{\xi}$)
-   **Rôle** : Encoder l'état futur $s_{t+1}$ pour fournir une cible stable.
-   **Mise à jour** : **EMA (Exponential Moving Average)** des poids de l'Online Encoder.
    -   $\xi \leftarrow \mu \xi + (1 - \mu) \theta$
    -   $\mu \approx 0.996$ (Momentum)
-   **Sortie** : $z_{t+1}^{target} = f_{\xi}(s_{t+1})$
-   **Pourquoi ?** L'utilisation d'une cible qui évolue lentement empêche le "Latent Collapse" (où tout s'effondre vers zéro) sans avoir besoin de "Negative Pairs" (Contrastive Learning).

### C. Predictor ($p_{\phi}$)
-   **Rôle** : Prédire la représentation future à partir de la présente et de l'action.
-   **Mise à jour** : Descente de gradient.
-   **Sortie** : $\hat{z}_{t+1} = p_{\phi}(z_t, a_t)$

## 3. Fonction de Perte (Loss)

L'objectif est de minimiser la distance entre la prédiction et la cible stable :

$$ \mathcal{L} = \| \hat{z}_{t+1} - z_{t+1}^{target} \|^2_2 $$

Nous ajoutons également une légère **Variance Regularization** pour accélérer la convergence et garantir que les dimensions latentes sont utilisées :

$$ \mathcal{L}_{var} = \text{ReLU}(1 - \sigma(z)) $$

## 4. Processus d'Entraînement

1.  **Sampling** : On tire un tuple $(s_t, a_t, s_{t+1})$ du dataset.
2.  **Forward Online** : $z_t$ et $\hat{z}_{t+1}$ sont calculés (avec gradients).
3.  **Forward Target** : $z_{t+1}^{target}$ est calculé (sans gradients).
4.  **Backward** : On minimise l'erreur de prédiction.
5.  **EMA Update** : On met à jour les poids du Target Encoder.

## 5. Utilisation
Une fois entraîné, l'**Online Encoder** et le **Predictor** peuvent être utilisés pour :
-   **Latent MPC** : Planifier des trajectoires.
-   **Policy Training** : Initialiser une policy avec un encodeur qui "comprend" déjà la physique du dessin.
