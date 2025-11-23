import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import yaml
from tqdm import tqdm
from datetime import datetime

from picassbot.policy.model import PolicyNetwork
from picassbot.policy.data import PolicyDataset

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device(config_device):
    """Get the appropriate device based on config."""
    if config_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using device: CUDA")
        else:
            device = torch.device("cpu")
            print("Using device: CPU")
    else:
        device = torch.device(config_device)
        print(f"Using device: {config_device}")
    return device

def train(config, args):
    # Device
    device = get_device(config['training']['device'])
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['training']['log_dir'], timestamp)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Log hyperparameters
    hparams = {
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs'],
        'weight_decay': config['training']['weight_decay'],
        'image_size': config['data']['image_size'],
        'hidden_dim': config['model']['hidden_dim'],
    }
    
    # Data
    print("Loading dataset...")
    categories = config['data']['categories']
    
    # Support for using all categories
    if categories == -1 or categories == "all" or (isinstance(categories, list) and len(categories) == 1 and categories[0] in [-1, "all"]):
        from picassbot.quickdraw import QuickDrawDataset
        qd = QuickDrawDataset(config['data']['data_dir'])
        categories = qd.categories
        print(f"Using ALL {len(categories)} categories")
    else:
        print(f"Using {len(categories)} categories: {categories}")
    
    dataset = PolicyDataset(
        config['data']['data_dir'], 
        categories, 
        max_samples=config['data']['max_drawings_per_category']
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Categories: {categories}")
    
    # Model
    model = PolicyNetwork(
        hidden_dim=config['model']['hidden_dim']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    
    model.train()
    
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        total_loss = 0
        total_cont_loss = 0
        total_eos_loss = 0
        total_eod_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, (canvas, target, action_gt) in enumerate(pbar):
            canvas = canvas.to(device)
            target = target.to(device)
            action_gt = action_gt.to(device)  # (B, 4) -> dx, dy, eos, eod
            
            # Forward
            mean, logstd, eos_logit, eod_logit = model(canvas, target)
            
            # Ground Truth
            dx_dy_gt = action_gt[:, 0:2]
            eos_gt = action_gt[:, 2:3]
            eod_gt = action_gt[:, 3:4]
            
            # 1. Continuous Loss (Negative Log Likelihood of Gaussian)
            std = torch.exp(logstd)
            var = std.pow(2)
            log_prob = -0.5 * ((dx_dy_gt - mean).pow(2) / var + 2 * logstd + torch.log(torch.tensor(2 * 3.14159)))
            loss_cont = -log_prob.mean() * config['training']['continuous_weight']
            
            # 2. Discrete Loss (EOS)
            loss_eos = bce_loss(eos_logit, eos_gt)
            
            # 3. Discrete Loss (EOD)
            loss_eod = bce_loss(eod_logit, eod_gt) * config['training']['discrete_weight']
            
            # Total Loss
            loss = loss_cont + loss_eos + loss_eod
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_cont_loss += loss_cont.item()
            total_eos_loss += loss_eos.item()
            total_eod_loss += loss_eod.item()
            
            # Log to TensorBoard
            if global_step % config['training']['log_every'] == 0:
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/continuous', loss_cont.item(), global_step)
                writer.add_scalar('Loss/eos', loss_eos.item(), global_step)
                writer.add_scalar('Loss/eod', loss_eod.item(), global_step)
                writer.add_scalar('Learning/lr', optimizer.param_groups[0]['lr'], global_step)
            
            pbar.set_postfix({
                "loss": loss.item(), 
                "cont": loss_cont.item(), 
                "eos": loss_eos.item(),
                "eod": loss_eod.item()
            })
            
            global_step += 1
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_cont = total_cont_loss / len(dataloader)
        avg_eos = total_eos_loss / len(dataloader)
        avg_eod = total_eod_loss / len(dataloader)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} (cont: {avg_cont:.4f}, eos: {avg_eos:.4f}, eod: {avg_eod:.4f})")
        
        # Log epoch averages
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        writer.add_scalar('Epoch/continuous', avg_cont, epoch)
        writer.add_scalar('Epoch/eos', avg_eos, epoch)
        writer.add_scalar('Epoch/eod', avg_eod, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
            checkpoint_path = os.path.join(config['training']['checkpoint_dir'], f"policy_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Log final hyperparameters with metrics
    writer.add_hparams(
        hparams,
        {
            'hparam/final_loss': avg_loss,
            'hparam/final_cont': avg_cont,
            'hparam/final_eos': avg_eos,
            'hparam/final_eod': avg_eod
        }
    )
    
    writer.close()
    print(f"\nTraining complete! TensorBoard logs saved to: {log_dir}")
    print(f"To view: tensorboard --logdir={config['training']['log_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--override", nargs='*', help="Override config values (e.g., training.batch_size=64)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        for override in args.override:
            keys, value = override.split('=')
            keys = keys.split('.')
            
            # Navigate to the right nested dict
            current = config
            for key in keys[:-1]:
                current = current[key]
            
            # Try to convert value to appropriate type
            try:
                value = eval(value)
            except:
                pass  # Keep as string
            
            current[keys[-1]] = value
            print(f"Override: {'.'.join(keys)} = {value}")
    
    train(config, args)
