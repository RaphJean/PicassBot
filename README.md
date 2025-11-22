# Picassbot 🎨🤖

Picassbot est un agent IA capable d'apprendre à dessiner comme un humain en utilisant le dataset QuickDraw. Il combine l'apprentissage par renforcement (Policy Learning), la planification (Latent MPC) et la modélisation du monde (JEPA).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RaphJean/PicassBot/blob/main/colab_training.ipynb)

## Fonctionnalités
- **Dessin Vectoriel** : Génère des séquences de traits (dx, dy, pen_up).
- **Planification** : Utilise des algorithmes comme MCTS et MPC pour planifier le dessin.
- **World Model (JEPA)** : Apprend la physique du dessin dans un espace latent pour une planification rapide.
- **Entraînement Joint** : Entraîne simultanément l'encodeur, le prédicteur et la policy.

## Installation

```bash
git clone https://github.com/RaphJean/PicassBot.git
cd PicassBot
pip install -r requirements.txt
```

## Utilisation

### Entraînement (Local)
```bash
# Entraînement Joint (Recommandé)
python -m policy.train_joint --config config.yaml

# Entraînement JEPA (Self-Supervised)
python -m policy.train_jepa --config config.yaml
```

### Entraînement (Gratuit sur GPU)
Cliquez sur le badge "Open in Colab" ci-dessus pour lancer l'entraînement gratuitement sur les serveurs de Google.

### Inférence / Démo
```bash
# Dessiner un carré avec Latent MPC
python -m research.run_experiments --strategy latent_mpc --target_type square --joint_model_path joint_checkpoints/last.pth
```
