#!/bin/bash
# Script to clean up old checkpoints and free disk space

echo "=== Disk Space Cleanup ==="
echo ""

# Check current disk usage
echo "Current disk usage:"
df -h . | tail -1
echo ""

# Find large files
echo "Large files in project:"
find . -type f -size +50M -not -path "./.venv/*" -not -path "./.git/*" 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  $size - $file"
done
echo ""

# Count checkpoints
policy_count=$(ls -1 policy_checkpoints/*.pth 2>/dev/null | wc -l)
predictor_count=$(ls -1 predictor_checkpoints/*.pth 2>/dev/null | wc -l)

echo "Checkpoints found:"
echo "  Policy: $policy_count"
echo "  Predictor: $predictor_count"
echo ""

# Offer to clean old checkpoints
if [ $policy_count -gt 3 ]; then
    echo "Keeping only the 3 most recent policy checkpoints..."
    ls -t policy_checkpoints/*.pth | tail -n +4 | xargs rm -f
    echo "  Removed $((policy_count - 3)) old policy checkpoints"
fi

if [ $predictor_count -gt 2 ]; then
    echo "Keeping only the 2 most recent predictor checkpoints..."
    ls -t predictor_checkpoints/*.pth | tail -n +3 | xargs rm -f
    echo "  Removed $((predictor_count - 2)) old predictor checkpoints"
fi

# Clean tensorboard logs older than 7 days
echo ""
echo "Cleaning old TensorBoard logs (>7 days)..."
find logs -type d -mtime +7 -not -path "logs" 2>/dev/null | xargs rm -rf
echo "  Done"

# Clean temporary files
echo ""
echo "Cleaning temporary files..."
find . -name "*.tmp" -o -name "*.pyc" -o -name "__pycache__" | xargs rm -rf
echo "  Done"

echo ""
echo "=== Cleanup Complete ==="
echo "New disk usage:"
df -h . | tail -1
