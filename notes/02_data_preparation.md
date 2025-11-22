# Étape 1 : Préparation des Données

## 1. Téléchargement des Données
Nous avons téléchargé des échantillons du dataset **Quick, Draw!** (format `.ndjson` simplifié) dans le dossier `data/raw/`.
Catégories actuelles :
*   `apple`
*   `face`
*   `car`

## 2. Librairie `src/quickdraw.py`
Une classe utilitaire `QuickDrawDataset` a été créée pour faciliter le chargement et la manipulation des données.

### Fonctionnalités :
*   **Chargement** : Lit les fichiers `.ndjson` et parse les lignes JSON.
*   **Visualisation** :
    *   `vector_to_image(drawing)` : Convertit un dessin vectoriel en image PNG (raster).
    *   `vector_to_raster_sequence(drawing)` : Génère une séquence d'images montrant la construction trait par trait (utile pour visualiser la dynamique temporelle).

## 3. Vérification
Le script `demo_quickdraw.py` a été exécuté avec succès. Il a généré des images de test dans le dossier `output_demo/`.
*   On confirme que les vecteurs sont bien lus.
*   La conversion en image fonctionne correctement.

## Prochaine Étape
Nous sommes prêts pour l'étape suivante : **Entraîner le Prédicteur (World Model)**.
Il faudra créer un dataset PyTorch qui génère des paires `(Image État T, Action, Image État T+1)` à la volée à partir de ces vecteurs.
