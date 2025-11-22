#!/usr/bin/env python
"""Joint training of encoder, predictor, and policy in FullAgent.
Trains both the world model (predictor) and the actor (policy) simultaneously.
"""

import os
import yaml
import torch
import torch.nn as nn
import tqdm
from policy.joint_model import FullAgent
from policy.joint_dataset import JointLatentDynamicsDataset
from policy.joint_dataset import JointLatentDynamicsDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw

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
    
    # 1. Create Target (Square)
    target_np = create_target_image("square")
    target = torch.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    target = target.to(device)
    
    # 2. Initialize Canvas (White)
    canvas_np = np.ones((128, 128), dtype=np.float32) * 255.0
    canvas = torch.from_numpy(canvas_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    canvas = canvas.to(device)
    
    # 3. Run Policy Loop
    images = []
    with torch.no_grad():
        # Pre-encode target once (optimization)
        z_target = model.encoder(target)
        
        for step in range(step_limit):
            # Encode current canvas
            z_curr = model.encoder(canvas)
            
            # Get action from policy
            mean, logstd, eos_logit, eod_logit = model.policy(z_curr, z_target)
            
            # Deterministic action
            dx_dy = mean
            eos = (torch.sigmoid(eos_logit) > 0.5).float()
            eod = (torch.sigmoid(eod_logit) > 0.5).float()
            
            # Apply action (Simple simulation for eval)
            # Note: This is a simplified "world" - just drawing lines
            # In a real scenario, we'd use the DrawingWorld engine, but here we approximate
            # to keep it self-contained or we can import the engine.
            # Let's use the engine if possible, but for now, let's just log the "attempt"
            # actually, without the engine, we can't update the canvas easily.
            # So let's just log the FIRST action and maybe the target.
            # BETTER: Let's use the real engine for one episode.
            break # Placeholder until we import engine
            
    # Re-implementation using DrawingWorld for accurate evaluation
    from picassbot.engine import DrawingWorld
    world = DrawingWorld(width=128, height=128)
    
    frames = []
    with torch.no_grad():
        z_target = model.encoder(target)
        
        for step in range(step_limit):
            state = world.get_state()
            frames.append(state)
            
            # Prepare input
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
                
    # 4. Log to TensorBoard
    if writer:
        # Log final canvas
        final_canvas = world.get_state()
        writer.add_image("Eval/FinalCanvas", final_canvas, epoch, dataformats='HW')
        writer.add_image("Eval/Target", target_np, epoch, dataformats='HW')
        
        # Compute MSE
        mse = np.mean((final_canvas - target_np) ** 2)
        writer.add_scalar("Eval/MSE", mse, epoch)
        
    model.train()


def train_joint(config, args):
    # Device selection
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = JointLatentDynamicsDataset(
        data_dir=config["data"]["data_dir"],
        categories=config["data"]["categories"],
        max_samples=config["data"].get("max_drawings_per_category", 100),
        action_dim=config["model"]["action_dim"],
    )
    print(f"Dataset size: {len(dataset)} transitions")
    
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=False,  # MPS doesn't support pin_memory
    )

    # Model - FullAgent with encoder, predictor, and policy
    model = FullAgent(
        action_dim=config["model"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"].get("weight_decay", 0.0)
    )
    
    # Loss functionsSi
    mse_loss = nn.MSELoss()
    
    # Loss weights
    predictor_weight = config["training"].get("predictor_weight", 1.0)
    policy_weight = config["training"].get("policy_weight", 1.0)
    variance_weight = config["training"].get("variance_weight", 1.0)  # Weight for variance regularization

    # Load pretrained policy if specified
    if args.pretrained_policy:
        model.load_pretrained_encoder(args.pretrained_policy)

    # TensorBoard
    log_dir = config["training"].get("log_dir", "logs/joint")
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
        epoch_predictor_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_total_loss = 0.0
        
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for canvas, target, next_canvas, action in pbar:
            canvas = canvas.to(device)
            target = target.to(device)
            next_canvas = next_canvas.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(canvas, target, next_canvas, action)
            
            # 1. Predictor loss (world model): predict next latent state
            predictor_loss = mse_loss(outputs["z_next_pred"], outputs["z_next_true"])
            
            # 2. Policy loss: predict action from current state and target
            action_mean, action_logstd, eos_logit, eod_logit = outputs["action_pred"]
            
            # Ground truth actions
            dx_dy_gt = action[:, :2]
            eos_gt = action[:, 2:3] if action.shape[1] >= 3 else torch.zeros((action.shape[0], 1), device=device)
            eod_gt = action[:, 3:4] if action.shape[1] >= 4 else torch.zeros((action.shape[0], 1), device=device)
            
            # Policy loss components:
            # A. Continuous action loss (negative log-likelihood of Gaussian)
            # Clamp logstd for numerical stability (already clamped in model, but double-check)
            action_logstd_clamped = torch.clamp(action_logstd, min=-20, max=2)
            std = torch.exp(action_logstd_clamped)
            var = std.pow(2) + 1e-6  # Add small epsilon for numerical stability
            
            # Compute negative log-likelihood
            log_prob = -0.5 * (
                (dx_dy_gt - action_mean).pow(2) / var 
                + 2 * action_logstd_clamped 
                + torch.log(torch.tensor(2 * 3.14159, device=device))
            )
            loss_continuous = -log_prob.mean()
            
            # B. EOS loss (binary classification)
            loss_eos = nn.BCEWithLogitsLoss()(eos_logit.squeeze(-1), eos_gt.squeeze(-1)) if action.shape[1] >= 3 else torch.tensor(0.0, device=device)
            
            # C. EOD loss (binary classification)  
            loss_eod = nn.BCEWithLogitsLoss()(eod_logit.squeeze(-1), eod_gt.squeeze(-1)) if action.shape[1] >= 4 else torch.tensor(0.0, device=device)
            
            # Total policy loss
            policy_loss = loss_continuous + loss_eos + loss_eod
            
            # Combined loss
            # Add variance regularization to prevent latent collapse
            # We want z to have variance across the batch (avoid all z being same)
            # Calculate std of latent vectors across batch
            z_curr = model.encoder(canvas) # Re-compute or grab from somewhere? 
            # Actually, we can't easily get z_curr from outputs without changing forward return
            # Let's assume outputs["z_next_true"] is a good proxy for latent distribution since it's also from encoder
            z_batch = outputs["z_next_true"] 
            z_std = torch.sqrt(z_batch.var(dim=0) + 1e-6)
            loss_variance = torch.relu(1.0 - z_std).mean() # Hinge loss: penalize if std < 1.0
            
            total_loss = predictor_weight * predictor_loss + policy_weight * policy_loss + variance_weight * loss_variance
            
            # Check for NaN before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n⚠️  NaN/Inf detected! pred={predictor_loss.item():.4f}, pol={policy_loss.item():.4f}, var={loss_variance.item():.4f}")
                print(f"   action_mean: min={action_mean.min():.4f}, max={action_mean.max():.4f}")
                print(f"   action_logstd: min={action_logstd.min():.4f}, max={action_logstd.max():.4f}")
                print(f"   Skipping this batch...")
                continue
            
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_predictor_loss += predictor_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_total_loss += total_loss.item()
            
            pbar.set_postfix({
                'pred': f'{predictor_loss.item():.4f}',
                'pol': f'{policy_loss.item():.4f}',
                'var': f'{loss_variance.item():.4f}',
                'z_std': f'{z_std.mean().item():.4f}'
            })
            
            if writer:
                global_step = epoch * len(loader) + pbar.n
                writer.add_scalar("Loss/predictor", predictor_loss.item(), global_step)
                writer.add_scalar("Loss/policy", policy_loss.item(), global_step)
                writer.add_scalar("Loss/variance", loss_variance.item(), global_step)
                writer.add_scalar("Loss/total", total_loss.item(), global_step)
                writer.add_scalar("Latent/std_mean", z_std.mean().item(), global_step)

        avg_predictor_loss = epoch_predictor_loss / len(loader)
        avg_policy_loss = epoch_policy_loss / len(loader)
        avg_total_loss = epoch_total_loss / len(loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Predictor Loss: {avg_predictor_loss:.6f}")
        print(f"  Policy Loss: {avg_policy_loss:.6f}")
        print(f"  Total Loss: {avg_total_loss:.6f}")
        
        if writer:
            writer.add_scalar("Loss/epoch_predictor", avg_predictor_loss, epoch+1)
            writer.add_scalar("Loss/epoch_policy", avg_policy_loss, epoch+1)
            writer.add_scalar("Loss/epoch_total", avg_total_loss, epoch+1)

        # Evaluate model
        if (epoch + 1) % config["eval"].get("eval_every", 1) == 0:
            print(f"Evaluating model at epoch {epoch+1}...")
            evaluate_model(model, device, writer, epoch+1)

        # Checkpoint (save encoder, predictor, and policy separately)
        if (epoch + 1) % config["training"].get("save_every", 1) == 0:
            ckpt = {
                "epoch": epoch + 1,
                "encoder_state_dict": model.encoder.state_dict(),
                "predictor_state_dict": model.predictor.state_dict(),
                "policy_state_dict": model.policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }
            ckpt_dir = config["training"].get("checkpoint_dir", "joint_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"joint_epoch_{epoch+1}.pth")
            
            # Use temp file to prevent corruption on disk space issues
            temp_path = ckpt_path + ".tmp"
            try:
                torch.save(ckpt, temp_path)
                # If successful, rename to final path
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                os.rename(temp_path, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                
                # Check file size
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                print(f"  Checkpoint size: {size_mb:.1f} MB")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
                print(f"  This may be due to insufficient disk space")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

    if writer:
        writer.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Joint encoder + predictor + policy training")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--pretrained_policy", default=None, help="Path to pretrained policy checkpoint to initialize encoder")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train_joint(cfg, args)
