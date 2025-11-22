#!/bin/bash
# Script to launch TensorBoard for monitoring training

echo "Starting TensorBoard..."
echo "Open http://localhost:6006 in your browser"
echo ""

tensorboard --logdir=logs/policy --port=6006
