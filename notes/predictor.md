# Architecture du Predictor

## Introduction
Le **Predictor** est un composant central de l'architecture JEPA (Joint Embedding Predictive Architecture). Son rôle est de modéliser la dynamique du monde dans l'espace latent. Concrètement, il doit prédire l'état latent futur $z_{t+1}$ à partir de l'état latent actuel $z_t$ et de l'action $a_t$.

Contrairement à un modèle génératif classique qui prédit des pixels (comme un VAE ou un GAN), le Predictor opère entièrement dans l'espace de représentation abstrait appris par l'encodeur. Cela lui permet de se concentrer sur la sémantique et la dynamique de haut niveau (ex: "tracer un trait vers la droite") plutôt que sur les détails de bas niveau (bruit de pixel, texture exacte).

Une bonne architecture de Predictor doit être capable de :
1.  Capturer les dépendances à long terme (mémoire).
2.  Gérer l'incertitude (stochastique vs déterministe).
3.  Être efficace en inférence (pour la planification rapide).

## Architecture Actuelle
Actuellement, nous utilisons une architecture récurrente classique (**LSTM** ou **GRU**).

### Implémentation
- **Entrée** : Concaténation de l'état latent $z_t$ et de l'action $a_t$.
- **Cœur** : Un réseau LSTM ou GRU à plusieurs couches (par défaut 2 couches, hidden_dim=512).
- **Sortie** : Une projection linéaire (MLP) de l'état caché du RNN vers l'espace latent $z_{t+1}$.

### Analyse
*   **Avantages** :
    *   Simple à implémenter et à entraîner.
    *   Efficace pour les séquences de longueur modérée.
    *   Maintient un état caché explicite qui sert de "mémoire".
*   **Limites** :
    *   **Déterministe** : L'implémentation actuelle est purement déterministe ($z_{t+1} = f(z_t, a_t)$). Elle ne peut pas modéliser plusieurs futurs possibles (multimodalité), ce qui est limitant pour des tâches créatives ou incertaines.
    *   **Oubli catastrophique** : Les LSTM peuvent avoir du mal sur de très longues séquences (> 100 pas) sans mécanismes d'attention.
    *   **Séquentiel** : L'entraînement n'est pas parallélisable sur la dimension temporelle (contrairement aux Transformers).

## Architectures Alternatives Intéressantes

Voici plusieurs architectures modernes qui pourraient améliorer les performances ou les capacités du modèle :

### 1. Transformer (GPT-style)
Utiliser un Transformer décodeur (causal) pour prédire la séquence d'états latents.
*   **Principe** : Traiter la séquence $(z_0, a_0, z_1, a_1, ...)$ comme des tokens.
*   **Avantages** :
    *   Meilleure gestion des dépendances à long terme grâce au mécanisme d'attention.
    *   Parallélisable à l'entraînement.
    *   Architecture standard très optimisée.
*   **Inconvénients** :
    *   Coût quadratique $O(T^2)$ en mémoire/calcul avec la longueur de la séquence (bien que gérable pour $T \approx 50-100$).
    *   Nécessite souvent plus de données.

### 2. RSSM (Recurrent State-Space Model) - Dreamer
L'architecture utilisée dans Dreamer (V1, V2, V3), l'état de l'art en Reinforcement Learning basé sur des modèles.
*   **Principe** : Décompose l'état en deux parties :
    *   **État déterministe ($h_t$)** : Géré par un GRU/RNN.
    *   **État stochastique ($s_t$)** : Échantillonné à partir d'une distribution (Gaussienne ou Catégorielle) paramétrée par $h_t$.
*   **Avantages** :
    *   **Stochastique** : Capable de modéliser l'incertitude et plusieurs futurs possibles. C'est crucial si l'environnement n'est pas parfaitement déterministe ou si l'encodeur perd de l'information.
    *   Très robuste.
*   **Recommandation** : C'est l'évolution la plus naturelle et puissante pour notre cas d'usage.

### 3. S4 / Mamba (Structured State Space Models)
Une nouvelle classe de modèles qui combine les avantages des RNNs (inférence rapide $O(1)$) et des Transformers (entraînement parallèle).
*   **Principe** : Utilise des équations d'état continues discrétisées efficacement.
*   **Avantages** :
    *   Performance comparable aux Transformers sur de longues séquences.
    *   Inférence très rapide (comme un RNN).
    *   Pas de coût quadratique.
*   **Inconvénients** :
    *   Plus complexe à implémenter (nécessite souvent des kernels CUDA spécifiques).

### 4. V-JEPA / I-JEPA (Vision Transformer based)
Les papiers originaux JEPA de Yann LeCun utilisent souvent des architectures basées sur des Vision Transformers (ViT).
*   **Principe** : Le prédicteur est un petit ViT qui prend en entrée les patchs masqués ou l'état latent et prédit les représentations manquantes.
*   **Pertinence** : Plus adapté si l'on travaille directement sur des patchs d'images ou des représentations spatiales, moins pour des vecteurs latents plats globaux.

## Recommandations

1.  **Court terme (Facile)** : Ajouter une composante **stochastique** à l'architecture actuelle (RSSM simplifié). Au lieu de prédire directement $z_{t+1}$, le RNN prédit les paramètres $(\mu, \sigma)$ d'une distribution gaussienne, et on échantillonne $z_{t+1} \sim \mathcal{N}(\mu, \sigma)$. Cela permet au modèle d'exprimer de l'incertitude.
2.  **Moyen terme (Performance)** : Passer à une architecture **Transformer** (type GPT-2 miniature) si la longueur des séquences reste raisonnable (< 200 pas). C'est souvent plus stable et performant que les LSTM classiques.
3.  **Long terme (SOTA)** : Implémenter un **RSSM complet (DreamerV3)**. C'est l'état de l'art actuel pour l'apprentissage de modèles du monde.
