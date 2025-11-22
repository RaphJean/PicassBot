# Recherche de Datasets : Dessin Vectoriel Généraliste

Ce document recense les bases de données potentielles pour entraîner un robot portraitiste. Nous cherchons idéalement des données **vectorielles** (séquences de traits) pour piloter le robot, et si possible des paires **Photo <-> Dessin** pour apprendre le style.

## 1. Quick, Draw! (Google)
*   **Description** : 50 millions de dessins réalisés en moins de 20 secondes par des utilisateurs du monde entier.
*   **Format** : `.ndjson` (Vectoriel : séquences de points x,y + temps).
*   **Contenu** : 345 catégories (objets, animaux, véhicules, plantes...).
*   **Avantages** :
    *   Format vectoriel natif avec information temporelle.
    *   **Diversité immense** : Couvre presque tous les objets du quotidien.
    *   Facile à utiliser.
*   **Inconvénients** :
    *   Qualité "gribouillage".
    *   Pas de photo associée (juste le label de classe).
*   **Verdict** : **La ressource #1** pour ce projet. La diversité des catégories est parfaite pour un robot généraliste.

## 2. The Sketchy Database
*   **Description** : Base de données de paires **Photo <-> Sketch**.
*   **Format** : Images (pixels) et **SVG** (Vectoriel). Contient l'information des traits.
*   **Contenu** : 12,500 images de 125 catégories.
*   **Avantages** :
    *   Données appariées (Photo <-> Dessin).
    *   **Multi-catégories** : Parfait pour généraliser au-delà des visages.
    *   Format vectoriel disponible.
*   **Inconvénients** :
    *   Moins de données brutes que Quick Draw.
*   **Verdict** : **Excellent complément** pour apprendre à relier une photo réelle à un sketch vectoriel.

## 3. APDrawing (Artist-Portrait Drawing)
*   **Description** : Dataset spécialisé pour la transformation de portraits en dessins au trait artistiques.
*   **Format** : Principalement Images (Pixels). Paires Photo / Dessin.
*   **Avantages** :
    *   Haute qualité artistique (portraits très beaux).
    *   Parfaitement aligné avec le but final (faire un beau portrait).
*   **Inconvénients** :
    *   **Pas vectoriel** (généralement). Il faudrait utiliser un algorithme de vectorisation (ex: `Potrace` ou un modèle de vectorisation) pour convertir les dessins en trajectoires robot.
    *   Pas d'ordre des traits (on perd le "comment c'est dessiné").
*   **Verdict** : **Moins pertinent** pour un robot généraliste, trop spécialisé "visage artistique".

## 4. TU-Berlin Sketch Dataset
*   **Description** : 20,000 croquis main levée.
*   **Format** : PNG et **SVG** (Vectoriel).
*   **Avantages** :
    *   Dessins humains réels.
    *   Format vectoriel.
*   **Inconvénients** :
    *   Pas de photos associées (juste des labels).
    *   Qualité variable.
*   **Verdict** : **Trop spécialisé**. À garder uniquement si on veut une "spécialisation portrait" plus tard.

## 5. FaceX / CUFS (CUHK Face Sketch)
*   **Description** : Datasets de recherche pour la reconnaissance faciale via croquis.
*   **Format** : Souvent Pixels.
*   **Avantages** :
    *   Focalisé 100% sur les visages.
    *   Paires Photo / Croquis.
*   **Inconvénients** :
    *   Souvent des croquis "identikit" ou très ombrés, difficiles à reproduire au trait simple par un robot.
    *   Rarement vectoriel natif.

---

## Recommandation Stratégique

1.  **Base Principale** : **Quick, Draw! (Toutes catégories)**. C'est le seul dataset qui offre la diversité nécessaire pour dessiner "n'importe quoi".
2.  **Apprentissage Visuel** : Utiliser **Sketchy** pour apprendre au modèle à extraire les traits importants d'une vraie photo (puisqu'il contient des paires Photo/Dessin).
3.  **Inférence** : Le robot prend une photo d'objet, V-JEPA l'analyse, et le planificateur utilise sa connaissance de Quick Draw pour reproduire l'objet.
