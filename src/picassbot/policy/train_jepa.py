#!/usr/bin/env python
"""
Self-Supervised Training for JEPA World Model.
Trains the encoder and predictor to model dynamics without any policy/task supervision.
"""

import os
import yaml
import torch
import torch.nn as nn
import tqdm
from picassbot.policy.jepa_model import JEPAWorldModel
from picassbot.policy.joint_dataset import JointLatentDynamicsDataset
from torch.utils.data import DataLoader

def train_jepa(config, args):
    # Device selection
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset
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
        pin_memory=False,
    )

    # Model
    model = JEPAWorldModel(
        action_dim=config["model"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        ema_momentum=config["training"].get("ema_momentum", 0.996)
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    
    # Loss
    mse_loss = nn.MSELoss()
    
    # TensorBoard
    log_dir = config["training"].get("log_dir", "logs/jepa")
    os.makedirs(log_dir, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    except Exception:
        pass

    epochs = config["training"]["num_epochs"]
    
    print("\n🚀 Starting JEPA Self-Supervised Training...")
    print(f"   EMA Momentum: {model.ema_momentum}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for canvas, _, next_canvas, action in pbar:
            canvas = canvas.to(device)
            next_canvas = next_canvas.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            
            # Forward
            z_next_pred, z_next_target = model(canvas, next_canvas, action)
            
            # Loss: Distance between prediction and stable target
            loss = mse_loss(z_next_pred, z_next_target)
            
            # Optional: Variance Regularization (Safety net against collapse)
            # Even with EMA, var reg helps convergence speed
            z_std = torch.sqrt(z_next_target.var(dim=0) + 1e-6)
            var_loss = torch.relu(1.0 - z_std).mean()
            
            total_loss = loss + 0.1 * var_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Update Target Encoder (EMA)
            model.update_target_encoder()

            epoch_loss += total_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'z_std': f'{z_std.mean().item():.4f}'
            })
            
            if writer:
                global_step = epoch * len(loader) + pbar.n
                writer.add_scalar("JEPA/loss", loss.item(), global_step)
                writer.add_scalar("JEPA/z_std", z_std.mean().item(), global_step)

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

        # Checkpoint
        if (epoch + 1) % config["training"].get("save_every", 1) == 0:
            ckpt_dir = config["training"].get("checkpoint_dir", "jepa_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(ckpt_dir, f"jepa_epoch_{epoch+1}.pth"))

    if writer:
        writer.close()
    print("\n✅ JEPA Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train_jepa(cfg, args)
