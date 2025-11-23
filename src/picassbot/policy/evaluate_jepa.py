#!/usr/bin/env python
"""
Evaluate JEPA World Model Quality

This script evaluates:
1. Encoder quality: Latent space variance, collapse detection
2. Predictor quality: Prediction accuracy (1-step and multi-step)
3. Visualization: Latent space and predictions
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
from picassbot.policy.jepa_model import JEPAWorldModel
from picassbot.policy.joint_model import FullAgent
from picassbot.policy.joint_dataset import JointLatentDynamicsDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_model(checkpoint_path, device):
    """Load JEPA model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Detect model type and dimensions
    is_jepa = 'online_encoder.net.0.conv.weight' in state_dict
    
    # Infer predictor hidden_dim
    predictor_hidden_dim = 256
    if 'predictor.lstm.weight_ih_l0' in state_dict:
        predictor_hidden_dim = state_dict['predictor.lstm.weight_ih_l0'].shape[0] // 4
    
    if is_jepa:
        print("Loading JEPAWorldModel...")
        model = JEPAWorldModel(action_dim=4, hidden_dim=predictor_hidden_dim)
        model.load_state_dict(state_dict)
    else:
        print("Loading FullAgent...")
        model = FullAgent(action_dim=4, hidden_dim=512, predictor_hidden_dim=predictor_hidden_dim)
        # Map encoder keys if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('online_encoder.'):
                new_key = k.replace('online_encoder.', 'encoder.')
                new_state_dict[new_key] = v
            elif k.startswith('predictor.'):
                new_state_dict[k] = v
            elif k.startswith('encoder.'):
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model, is_jepa


def evaluate_encoder_quality(model, loader, device, is_jepa=True):
    """Evaluate encoder quality: variance, collapse detection."""
    print("\n" + "="*60)
    print("ENCODER QUALITY EVALUATION")
    print("="*60)
    
    all_latents = []
    
    with torch.no_grad():
        for i, (canvas, target, next_canvas, action) in enumerate(loader):
            if i >= 50:  # Sample 50 batches
                break
            
            canvas = canvas.to(device)
            
            if is_jepa:
                z = model.online_encoder(canvas)
            else:
                z = model.encoder(canvas)
            
            all_latents.append(z.cpu().numpy())
    
    all_latents = np.concatenate(all_latents, axis=0)  # [N, latent_dim]
    
    # Metrics
    mean = all_latents.mean(axis=0)
    std = all_latents.std(axis=0)
    
    print(f"\n📊 Latent Statistics:")
    print(f"   Shape: {all_latents.shape}")
    print(f"   Mean (across batch): {mean.mean():.4f} ± {mean.std():.4f}")
    print(f"   Std (across batch):  {std.mean():.4f} ± {std.std():.4f}")
    
    # Collapse detection
    collapsed_dims = (std < 0.01).sum()
    collapse_ratio = collapsed_dims / len(std)
    
    print(f"\n⚠️  Collapse Detection:")
    print(f"   Collapsed dimensions (std < 0.01): {collapsed_dims} / {len(std)} ({collapse_ratio*100:.1f}%)")
    
    if collapse_ratio > 0.5:
        print("   ❌ SEVERE COLLAPSE: Over 50% of latent dimensions have collapsed!")
    elif collapse_ratio > 0.2:
        print("   ⚠️  WARNING: Significant collapse detected (>20%)")
    else:
        print("   ✅ Healthy latent space")
    
    # Effective rank (measure of dimensionality usage)
    cov = np.cov(all_latents.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    eigenvalues = eigenvalues / eigenvalues.sum()
    effective_rank = np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-10)))
    
    print(f"\n📐 Effective Rank: {effective_rank:.1f} / {all_latents.shape[1]}")
    print(f"   (Higher is better, max = {all_latents.shape[1]})")
    
    return {
        'mean': mean.mean(),
        'std': std.mean(),
        'collapse_ratio': collapse_ratio,
        'effective_rank': effective_rank
    }


def evaluate_predictor_quality(model, loader, device, is_jepa=True, max_horizon=5):
    """Evaluate predictor quality: 1-step and multi-step prediction accuracy."""
    print("\n" + "="*60)
    print("PREDICTOR QUALITY EVALUATION")
    print("="*60)
    
    one_step_errors = []
    multi_step_errors = {h: [] for h in range(1, max_horizon + 1)}
    
    with torch.no_grad():
        for i, (canvas, target, next_canvas, action) in enumerate(loader):
            if i >= 50:  # Sample 50 batches
                break
            
            canvas = canvas.to(device)
            next_canvas = next_canvas.to(device)
            action = action.to(device)
            
            # Get encoder
            if is_jepa:
                z_curr = model.online_encoder(canvas)
                z_next_true = model.target_encoder(next_canvas)
            else:
                z_curr = model.encoder(canvas)
                z_next_true = model.encoder(next_canvas)
            
            # 1-step prediction
            z_next_pred, _ = model.predictor(z_curr, action)
            error = torch.nn.functional.mse_loss(z_next_pred, z_next_true)
            one_step_errors.append(error.item())
            
            # Multi-step prediction (rollout)
            z = z_curr.clone()
            for h in range(1, max_horizon + 1):
                z, _ = model.predictor(z, action)
                error_h = torch.nn.functional.mse_loss(z, z_next_true)
                multi_step_errors[h].append(error_h.item())
    
    # Results
    print(f"\n📈 Prediction Errors (MSE):")
    print(f"   1-step:  {np.mean(one_step_errors):.6f} ± {np.std(one_step_errors):.6f}")
    
    print(f"\n   Multi-step errors:")
    for h in range(1, max_horizon + 1):
        mean_error = np.mean(multi_step_errors[h])
        std_error = np.std(multi_step_errors[h])
        print(f"   {h}-step:  {mean_error:.6f} ± {std_error:.6f}")
    
    # Quality assessment
    one_step_mean = np.mean(one_step_errors)
    if one_step_mean < 0.01:
        print(f"\n   ✅ Excellent prediction quality (MSE < 0.01)")
    elif one_step_mean < 0.1:
        print(f"\n   ✅ Good prediction quality (MSE < 0.1)")
    elif one_step_mean < 1.0:
        print(f"\n   ⚠️  Moderate prediction quality (MSE < 1.0)")
    else:
        print(f"\n   ❌ Poor prediction quality (MSE > 1.0)")
    
    return {
        'one_step_mse': one_step_mean,
        'multi_step_mse': {h: np.mean(multi_step_errors[h]) for h in range(1, max_horizon + 1)}
    }


def visualize_predictions(model, loader, device, is_jepa=True, save_path='jepa_eval_predictions.png'):
    """Visualize prediction errors over horizons."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    max_horizon = 10
    errors_per_horizon = {h: [] for h in range(1, max_horizon + 1)}
    
    with torch.no_grad():
        for i, (canvas, target, next_canvas, action) in enumerate(loader):
            if i >= 20:
                break
            
            canvas = canvas.to(device)
            next_canvas = next_canvas.to(device)
            action = action.to(device)
            
            if is_jepa:
                z_curr = model.online_encoder(canvas)
                z_target = model.target_encoder(next_canvas)
            else:
                z_curr = model.encoder(canvas)
                z_target = model.encoder(next_canvas)
            
            z = z_curr.clone()
            for h in range(1, max_horizon + 1):
                z, _ = model.predictor(z, action)
                error = torch.nn.functional.mse_loss(z, z_target, reduction='none').mean(dim=1)
                errors_per_horizon[h].extend(error.cpu().numpy())
    
    # Plot
    plt.figure(figsize=(10, 6))
    horizons = list(range(1, max_horizon + 1))
    mean_errors = [np.mean(errors_per_horizon[h]) for h in horizons]
    std_errors = [np.std(errors_per_horizon[h]) for h in horizons]
    
    plt.plot(horizons, mean_errors, 'o-', linewidth=2, markersize=8, label='Mean MSE')
    plt.fill_between(horizons, 
                     np.array(mean_errors) - np.array(std_errors),
                     np.array(mean_errors) + np.array(std_errors),
                     alpha=0.3, label='± 1 std')
    
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('JEPA Predictor: Multi-Step Prediction Error', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Saved visualization to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate JEPA World Model Quality")
    parser.add_argument("--checkpoint", required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--max_horizon", type=int, default=5, help="Max prediction horizon to test")
    args = parser.parse_args()
    
    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Load config
    config = yaml.safe_load(open(args.config))
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, is_jepa = load_model(args.checkpoint, device)
    
    # Load dataset
    print("\nLoading dataset for evaluation...")
    dataset = JointLatentDynamicsDataset(
        data_dir=config["data"]["data_dir"],
        categories=config["data"]["categories"],
        max_samples=config["data"].get("max_drawings_per_category", 1000),
        action_dim=config["model"]["action_dim"],
    )
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Evaluate
    encoder_metrics = evaluate_encoder_quality(model, loader, device, is_jepa)
    predictor_metrics = evaluate_predictor_quality(model, loader, device, is_jepa, args.max_horizon)
    
    # Visualize
    save_path = f"jepa_eval_{Path(args.checkpoint).stem}.png"
    visualize_predictions(model, loader, device, is_jepa, save_path)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n📊 Encoder:")
    print(f"   Collapse Ratio: {encoder_metrics['collapse_ratio']*100:.1f}%")
    print(f"   Effective Rank: {encoder_metrics['effective_rank']:.1f}")
    
    print(f"\n📈 Predictor:")
    print(f"   1-step MSE: {predictor_metrics['one_step_mse']:.6f}")
    print(f"   {args.max_horizon}-step MSE: {predictor_metrics['multi_step_mse'][args.max_horizon]:.6f}")
    
    # Overall grade
    grade = "✅ EXCELLENT" if (encoder_metrics['collapse_ratio'] < 0.2 and 
                             predictor_metrics['one_step_mse'] < 0.1) else \
            "✅ GOOD" if (encoder_metrics['collapse_ratio'] < 0.4 and 
                         predictor_metrics['one_step_mse'] < 0.5) else \
            "⚠️  NEEDS IMPROVEMENT"
    
    print(f"\n🎯 Overall Quality: {grade}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
