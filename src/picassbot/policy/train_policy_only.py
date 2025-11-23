#!/usr/bin/env python
"""Train only the policy head using pretrained JEPA encoder/predictor.

This script:
1. Loads pretrained encoder + predictor from a JEPA checkpoint
2. Freezes their weights
3. Trains only the policy head
4. Saves the complete FullAgent (encoder + predictor + policy) for use with LatentMPC
"""

import os
import yaml
import torch
import torch.nn as nn
import tqdm
from picassbot.policy.joint_model import FullAgent, PolicyHead
from picassbot.policy.data import PolicyDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import argparse


def create_target_image(shape_type="square", width=128, height=128):
    """Create a simple target image for evaluation."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    if shape_type == "square":
        draw.rectangle([30, 30, 98, 98], outline=0, width=2)
    elif shape_type == "circle":
        draw.ellipse([30, 30, 98, 98], outline=0, width=2)
        
    return np.array(img)


def evaluate_model(model, device, writer, epoch, step_limit=20):
    """Run a quick evaluation: draw a square and log to TensorBoard."""
    model.eval()
    
    from picassbot.engine import DrawingWorld
    world = DrawingWorld(width=128, height=128)
    
    target_np = create_target_image("square")
    target = torch.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    target = target.to(device)
    
    frames = []
    with torch.no_grad():
        z_target = model.encoder(target)
        
        for step in range(step_limit):
            state = world.get_state()
            frames.append(state)
            
            canvas = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) / 255.0
            canvas = canvas.to(device)
            
            z_curr = model.encoder(canvas)
            mean, _, eos_logit, eod_logit = model.policy(z_curr, z_target)
            
            dx, dy = mean[0, 0].item(), mean[0, 1].item()
            eos = 1.0 if torch.sigmoid(eos_logit).item() > 0.5 else 0.0
            eod = 1.0 if torch.sigmoid(eod_logit).item() > 0.5 else 0.0
            
            action = np.array([dx, dy, eos, eod])
            world.step(action)
            
            if eod > 0.5:
                break
                
    if writer:
        final_canvas = world.get_state()
        writer.add_image("Eval/FinalCanvas", final_canvas, epoch, dataformats='HW')
        writer.add_image("Eval/Target", target_np, epoch, dataformats='HW')
        
        mse = np.mean((final_canvas - target_np) ** 2)
        writer.add_scalar("Eval/MSE", mse, epoch)
        
    model.train()


def train_policy_only(config, args):
    """Train only the policy head with frozen JEPA encoder/predictor."""
    
    # Device selection
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset (standard policy dataset for action prediction)
    dataset = PolicyDataset(
        data_dir=config["data"]["data_dir"],
        categories=config["data"]["categories"],
        max_samples=config["data"].get("max_drawings_per_category", 1000),
        img_size=config["data"].get("image_size", 128),
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    # Load JEPA checkpoint
    if not args.jepa_checkpoint:
        raise ValueError("--jepa_checkpoint is required to load pretrained encoder/predictor")
    
    print(f"Loading JEPA checkpoint from: {args.jepa_checkpoint}")
    jepa_ckpt = torch.load(args.jepa_checkpoint, map_location='cpu')
    
    # Infer predictor hidden_dim from checkpoint
    state_dict = jepa_ckpt.get('model_state_dict', jepa_ckpt)
    predictor_hidden_dim = 256  # default
    if 'predictor.lstm.weight_ih_l0' in state_dict:
        predictor_hidden_dim = state_dict['predictor.lstm.weight_ih_l0'].shape[0] // 4
    elif 'predictor.output_proj.0.weight' in state_dict:
        predictor_hidden_dim = state_dict['predictor.output_proj.0.weight'].shape[0]
    
    print(f"Inferred predictor_hidden_dim: {predictor_hidden_dim}")
    
    # Create FullAgent with matching dimensions
    model = FullAgent(
        action_dim=config["model"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        predictor_hidden_dim=predictor_hidden_dim
    ).to(device)

    # Load encoder and predictor from JEPA checkpoint
    # Handle both JEPA (online_encoder) and FullAgent (encoder) formats
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('online_encoder.'):
            new_key = k.replace('online_encoder.', 'encoder.')
            new_state_dict[new_key] = v
        elif k.startswith('predictor.'):
            new_state_dict[k] = v
        elif k.startswith('encoder.'):
            new_state_dict[k] = v
    
    # Load with strict=False to allow missing policy weights
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Loaded pretrained encoder and predictor from JEPA checkpoint")
    
    # Freeze encoder and predictor
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.predictor.parameters():
        param.requires_grad = False
    
    print("🔒 Froze encoder and predictor weights")
    print("🎯 Training only the policy head")
    
    # Only optimize policy parameters
    optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0)
    )
    
    # TensorBoard
    log_dir = config["training"].get("log_dir", "logs/policy_only")
    os.makedirs(log_dir, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    except Exception:
        print("TensorBoard not available")

    epochs = config["training"]["num_epochs"]
    for epoch in range(epochs):
        model.train()
        # Keep encoder/predictor in eval mode
        model.encoder.eval()
        model.predictor.eval()
        
        epoch_loss = 0.0
        
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for canvas, target, action_gt in pbar:
            canvas = canvas.to(device)
            target = target.to(device)
            action_gt = action_gt.to(device)

            optimizer.zero_grad()
            
            # Forward pass (no gradients for encoder)
            with torch.no_grad():
                z_curr = model.encoder(canvas)
                z_targ = model.encoder(target)
            
            # Policy forward (with gradients)
            action_mean, action_logstd, eos_logit, eod_logit = model.policy(z_curr, z_targ)
            
            # Ground truth actions
            dx_dy_gt = action_gt[:, :2]
            eos_gt = action_gt[:, 2:3] if action_gt.shape[1] >= 3 else torch.zeros((action_gt.shape[0], 1), device=device)
            eod_gt = action_gt[:, 3:4] if action_gt.shape[1] >= 4 else torch.zeros((action_gt.shape[0], 1), device=device)
            
            # Policy loss components
            action_logstd_clamped = torch.clamp(action_logstd, min=-20, max=2)
            std = torch.exp(action_logstd_clamped)
            var = std.pow(2) + 1e-6
            
            # Negative log-likelihood
            log_prob = -0.5 * (
                (dx_dy_gt - action_mean).pow(2) / var 
                + 2 * action_logstd_clamped 
                + torch.log(torch.tensor(2 * 3.14159, device=device))
            )
            loss_continuous = -log_prob.mean()
            
            # Binary classification losses
            loss_eos = nn.BCEWithLogitsLoss()(eos_logit.squeeze(-1), eos_gt.squeeze(-1)) if action_gt.shape[1] >= 3 else torch.tensor(0.0, device=device)
            loss_eod = nn.BCEWithLogitsLoss()(eod_logit.squeeze(-1), eod_gt.squeeze(-1)) if action_gt.shape[1] >= 4 else torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = (
                config["training"].get("continuous_weight", 1.0) * loss_continuous +
                config["training"].get("discrete_weight", 1.0) * (loss_eos + loss_eod)
            )
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n⚠️  NaN/Inf detected! Skipping batch...")
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'cont': f'{loss_continuous.item():.4f}',
                'eos': f'{loss_eos.item():.4f}',
                'eod': f'{loss_eod.item():.4f}'
            })
            
            if writer:
                global_step = epoch * len(loader) + pbar.n
                writer.add_scalar("Loss/total", total_loss.item(), global_step)
                writer.add_scalar("Loss/continuous", loss_continuous.item(), global_step)
                writer.add_scalar("Loss/eos", loss_eos.item(), global_step)
                writer.add_scalar("Loss/eod", loss_eod.item(), global_step)

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1} Summary: Avg Loss: {avg_loss:.6f}")
        
        if writer:
            writer.add_scalar("Loss/epoch", avg_loss, epoch+1)

        # Evaluate
        if (epoch + 1) % config["eval"].get("eval_every", 1) == 0:
            print(f"Evaluating model at epoch {epoch+1}...")
            evaluate_model(model, device, writer, epoch+1)

        # Save checkpoint (complete FullAgent: encoder + predictor + policy)
        if (epoch + 1) % config["training"].get("save_every", 1) == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),  # Save complete model
                "encoder_state_dict": model.encoder.state_dict(),
                "predictor_state_dict": model.predictor.state_dict(),
                "policy_state_dict": model.policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }
            ckpt_dir = config["training"].get("checkpoint_dir", "policy_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"full_agent_epoch_{epoch+1}.pth")
            
            torch.save(ckpt, ckpt_path)
            print(f"✅ Saved complete FullAgent checkpoint: {ckpt_path}")
            
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            print(f"   Checkpoint size: {size_mb:.1f} MB")

    if writer:
        writer.close()
    print("\n✅ Training complete! Your FullAgent is ready for LatentMPC.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train policy head only with frozen JEPA encoder/predictor")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--jepa_checkpoint", required=True, help="Path to pretrained JEPA checkpoint")
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    train_policy_only(cfg, args)
