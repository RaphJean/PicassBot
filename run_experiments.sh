#!/bin/bash

# Script to run various planning experiments with Picassbot

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Define checkpoint paths (Update these if you have better checkpoints)
POLICY_CHECKPOINT="policy_checkpoints/policy_epoch_2.pth"
JOINT_CHECKPOINT="policy_checkpoints/joint_epoch_1.pth" 
# Note: Ensure joint_epoch_2.pth matches the architecture defined in code (hidden_dim=512 etc.)
# If you get size mismatch errors, try retraining or checking config.yaml

echo "=================================================="
echo "Picassbot Experiment Runner"
echo "=================================================="

# 1. Greedy Search (Baseline)
echo ""
echo "1. Running Greedy Search (Baseline)..."
echo "Target: Triangle"
python -m picassbot.planning.run_experiments \
    --strategy greedy \
    --target_type triangle \
    --model_path "$POLICY_CHECKPOINT"

# 2. MCTS (Monte Carlo Tree Search)
echo ""
echo "2. Running MCTS..."
echo "Target: Square"
python -m picassbot.planning.run_experiments \
    --strategy mcts \
    --target_type square \
    --mcts_simulations 50 \
    --model_path "$POLICY_CHECKPOINT"

# 3. Latent MPC (Requires Joint Model)
if [ -f "$JOINT_CHECKPOINT" ]; then
    echo ""
    echo "3. Running Latent MPC (JEPA)..."
    echo "Target: Circle"
    python -m picassbot.planning.run_experiments \
        --strategy latent_mpc \
        --target_type circle \
        --joint_model_path "$JOINT_CHECKPOINT" \
        --horizon 5 \
        --num_sequences 20
else
    echo ""
    echo "Skipping Latent MPC: Checkpoint $JOINT_CHECKPOINT not found."
fi

# 4. Dataset Target (Cat)
echo ""
echo "4. Running Greedy on Dataset Target (Cat)..."
python -m picassbot.planning.run_experiments \
    --strategy greedy \
    --target_type dataset \
    --target_category cat \
    --target_index 0 \
    --model_path "$POLICY_CHECKPOINT"

echo ""
echo "=================================================="
echo "Experiments completed. Results in planning_output/"
echo "=================================================="
