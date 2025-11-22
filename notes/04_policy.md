# Stratégies pour le Réseau de Politique (Policy Network)

Ce document explore les différentes approches possibles pour entraîner un réseau de neurones capable de dessiner (`Policy Network`).

## 1. Imitation Learning (Behavioral Cloning) - *Approche Actuelle*

C'est la stratégie la plus directe pour démarrer. On apprend au robot à imiter les traits humains du dataset QuickDraw.

*   **Principe** : $\pi(a_t | s_t, I_{target}) \approx \text{Expert}(s_t, I_{target})$
*   **Entrées** : État courant du canevas ($S_t$) + Image cible finale ($I_{target}$).
*   **Sortie** : Action ($dx, dy, eos, eod$).
*   **Avantages** :
    *   Entraînement stable et rapide (Supervised Learning).
    *   Produit des dessins "humains" (ordre des traits logique).
*   **Inconvénients** :
    *   **Distribution Shift** : Si le robot fait une erreur, il se retrouve dans un état jamais vu dans le dataset et ne sait pas comment corriger (problème classique du Behavioral Cloning).
    *   Nécessite des données annotées (séquences de traits).

## 2. Goal-Conditioned Reinforcement Learning (RL)

On entraîne un agent à maximiser une récompense (similitude avec l'image cible) par essais-erreurs.

*   **Algorithmes** : PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic).
*   **Récompense** : $R = - || \text{Canvas} - I_{target} ||^2$ (MSE) ou similarité perceptuelle (LPIPS, CLIP).
*   **Avantages** :
    *   Peut découvrir des stratégies de dessin originales (non humaines).
    *   Robuste aux erreurs (apprend à corriger).
    *   Ne nécessite pas de séquences de traits (juste l'image finale).
*   **Inconvénients** :
    *   Très difficile à entraîner (Reward Sparse, espace d'action continu).
    *   Lent à converger.

## 3. Inverse Dynamics Model (Action Inference)

Au lieu de prédire l'action directement, on apprend à prédire l'action qui a causé une transition d'état.

*   **Principe** : $P(a_t | s_t, s_{t+1})$
*   **Utilisation** :
    1.  Un planificateur (ou un modèle génératif) imagine l'état suivant souhaité $s_{t+1}$.
    2.  Le modèle inverse déduit l'action motrice pour y arriver.
*   **Avantages** :
    *   Découplage entre la "vision" (quoi dessiner) et le "contrôle" (comment bouger le bras).
    *   Réutilisable pour d'autres tâches.

## 4. Residual Policy (Correction de Planificateur)

On utilise un planificateur classique (ex: Greedy ou MCTS) pour proposer une action de base, et le réseau de neurones apprend à *corriger* cette action.

*   **Principe** : $a_{final} = a_{planner} + \pi_{residual}(s_t, I_{target})$
*   **Avantages** :
    *   Combine la robustesse du planificateur et l'intuition du réseau.
    *   Apprentissage plus facile (le réseau n'a qu'à apprendre le "delta").

## 5. Latent Space Planning (JEPA / World Model)

C'est l'objectif final du projet. On ne planifie pas dans l'espace des pixels (trop complexe), mais dans un espace latent abstrait.

*   **Architecture** :
    *   **Encoder** : $s_t \rightarrow z_t$
    *   **Predictor** : $z_{t+1} = P(z_t, a_t)$
    *   **Policy** : Recherche dans l'espace latent pour trouver la séquence $a_{0:T}$ qui minimise la distance avec $z_{target}$.
*   **Avantages** :
    *   Ignore les détails inutiles (bruit de pixel).
    *   Planification hiérarchique possible.
    *   Très efficace pour le raisonnement à long terme.

---

**Recommandation** : Commencer par l'approche **1 (Imitation Learning)** pour avoir une "baseline" solide et un agent capable de dessiner correctement. Ensuite, utiliser ce réseau comme "prior" pour guider une recherche **MCTS** ou évoluer vers l'approche **5 (JEPA)** pour la planification complexe.
