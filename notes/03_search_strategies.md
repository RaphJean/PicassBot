# Stratégies de Recherche : De la Feuille Blanche au Dessin Final

**Objectif** : Trouver une séquence d'actions $a_1, ..., a_T$ qui transforme l'état initial $S_0$ (feuille blanche) en un état final $S_T$ qui ressemble à l'image cible $I_{target}$.

Nous supposons disposer d'un **World Model (JEPA)** capable de prédire l'état futur (dans l'espace latent) étant donné un état actuel et une action : $\hat{s}_{t+1} = P(s_t, a_t)$.
Nous avons aussi une fonction de coût (Loss) dans l'espace latent : $L(s, s_{target})$.

Voici les différentes stratégies possibles pour le "Cerveau" du robot :

## 1. Recherche Gloutonne (Greedy Search)
*   **Principe** : À chaque étape $t$, on teste $N$ actions aléatoires (ou une grille d'actions). On choisit celle qui minimise immédiatement la distance avec la cible.
*   **Avantages** : Très simple, rapide.
*   **Inconvénients** : Myope. Peut se coincer dans des minimums locaux (ex: dessiner un trait qui semble bon maintenant mais empêche de finir le dessin correctement). Ne planifie pas.

## 2. Recherche par Faisceau (Beam Search)
*   **Principe** : Au lieu de garder seulement le meilleur état, on garde les $K$ meilleures séquences d'actions (le "faisceau"). À chaque étape, on étend ces $K$ séquences et on ne garde que les $K$ meilleures résultantes.
*   **Avantages** : Explore mieux que le glouton.
*   **Inconvénients** : Coûteux en mémoire et calcul si $K$ est grand. Reste limité pour la planification à très long terme.

## 3. MPC (Model Predictive Control) avec Random Shooting / CEM
*   **C'est la méthode classique avec les World Models (ex: I-JEPA).**
*   **Principe** :
    1.  À l'étape $t$, on génère $N$ séquences d'actions aléatoires de longueur $H$ (horizon, ex: 5 traits).
    2.  On utilise le World Model pour simuler l'état final de ces séquences.
    3.  On évalue la distance avec la cible.
    4.  On choisit la première action de la meilleure séquence.
    5.  On exécute cette action, et on recommence.
*   **Variante CEM (Cross-Entropy Method)** : Au lieu de tirer au hasard à chaque fois, on affine la distribution des actions (moyenne, variance) vers les meilleures séquences trouvées.
*   **Avantages** : Robuste, planifie à moyen terme, ne nécessite pas de gradient.

## 4. MCTS (Monte Carlo Tree Search)
*   **Principe** : Construire un arbre de recherche asymétrique. On explore beaucoup les branches prometteuses et un peu les autres.
*   **Avantages** : Excellent pour la planification stratégique à long terme (comme aux échecs/Go).
*   **Inconvénients** : Très lourd en calcul. Peut être overkill pour du dessin simple, mais intéressant pour des dessins complexes (ex: commencer par l'esquisse globale avant les détails).

## 5. Optimisation par Gradient (Differentiable Rendering)
*   **Principe** : Si notre World Model (ou le moteur de rendu) est différentiable, on peut considérer les paramètres de l'action (dx, dy) comme des variables à optimiser.
*   **Méthode** : On initialise une séquence d'actions aléatoires, puis on fait une descente de gradient pour minimiser la Loss finale.
*   **Avantages** : Peut trouver des solutions très précises et fluides.
*   **Inconvénients** : Nécessite que tout soit différentiable. Risque de "Vanishing Gradient" sur de longues séquences.

## 6. Apprentissage par Renforcement (RL - Policy Learning)
*   **Principe** : Entraîner un réseau de neurones (l'Agent) qui prend l'état en entrée et sort directement l'action optimale.
*   **Entraînement** : On utilise le World Model comme simulateur pour entraîner cet agent (Dreamer, PPO, etc.).
*   **Avantages** : Une fois entraîné, l'inférence est instantanée (pas de recherche coûteuse au moment du dessin).
*   **Inconvénients** : Long et difficile à entraîner. Moins flexible si on change de type de dessin.

---

## Recommandation pour le Projet

Commencer par **3. MPC avec Random Shooting (ou CEM)**.
*   C'est la méthode naturelle pour les JEPA.
*   C'est flexible (on peut changer l'horizon, le nombre de samples).
*   Ça permet de valider le World Model rapidement.

Si le MPC est trop lent ou imprécis, on pourra envisager d'entraîner une Policy (RL) par la suite.
