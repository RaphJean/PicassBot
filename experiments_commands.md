# Commandes pour lancer les expériences

## 0. Entraînement

### Entraîner seulement la Policy (avec JEPA pré-entraîné) - RECOMMANDÉ
Cette commande entraîne **seulement** la tête de policy en utilisant l'encoder et le predictor déjà entraînés dans JEPA.
```bash
python -m picassbot.policy.train_policy_only --config config.yaml --jepa_checkpoint policy_checkpoints/jepa_epoch_2.pth
```
Le checkpoint final sera sauvegardé dans `policy_checkpoints/full_agent_epoch_X.pth` et sera **directement utilisable** avec LatentMPC.

### Entraîner un FullAgent complet (encoder + predictor + policy)
```bash
python -m picassbot.policy.train_joint --config config.yaml
```

### Évaluer la qualité de JEPA (Encodeur + Prédicteur)
```bash
python -m picassbot.policy.evaluate_jepa --checkpoint policy_checkpoints/jepa_epoch_4.pth --config config.yaml
```
Génère un rapport détaillé et une visualisation des erreurs de prédiction.



Voici une liste de commandes prêtes à l'emploi. Assurez-vous d'être à la racine du projet et que votre environnement virtuel est activé.

## 1. Baselines (Sans modèle joint)

### Greedy Search (Le plus rapide)
Dessiner un triangle :
```bash
python -m picassbot.planning.run_experiments --strategy greedy --target_type triangle --model_path policy_checkpoints/policy_epoch_2.pth
```

### MCTS (Monte Carlo Tree Search)
Dessiner un carré (plus lent mais meilleure planification) :
```bash
python -m picassbot.planning.run_experiments --strategy mcts --target_type square --mcts_simulations 50 --model_path policy_checkpoints/policy_epoch_2.pth
```

## 2. Latent MPC (Avec JEPA/FullAgent)

C'est la méthode principale qui utilise le modèle du monde.

### Dessiner un cercle
```bash
python -m picassbot.planning.run_experiments --strategy latent_mpc --target_type circle --joint_model_path policy_checkpoints/jepa_epoch_2.pth --horizon 5 --num_sequences 20
```

### Dessiner un carré
```bash
python -m picassbot.planning.run_experiments --strategy latent_mpc --target_type square --joint_model_path policy_checkpoints/joint_epoch_2.pth --horizon 5 --num_sequences 20
```

## 3. Sur le Dataset QuickDraw

### Reproduire un Chat (Cat)
```bash
python -m picassbot.planning.run_experiments --strategy latent_mpc --target_type dataset --target_category cat --target_index 0 --joint_model_path policy_checkpoints/joint_epoch_2.pth
```

### Reproduire un Visage (Face)
```bash
python -m picassbot.planning.run_experiments --strategy latent_mpc --target_type dataset --target_category face --target_index 0 --joint_model_path policy_checkpoints/joint_epoch_2.pth
```

## 4. Comparaison de toutes les stratégies
Lance toutes les stratégies sur la même cible pour comparer :
```bash
python -m picassbot.planning.run_experiments --strategy all --target_type triangle --model_path policy_checkpoints/policy_epoch_2.pth --joint_model_path policy_checkpoints/joint_epoch_2.pth
```
