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

from policy.predictor import LatentPredictor, LatentPredictorGRU
from policy.predictor_data import SimpleLatentDynamicsDataset
from policy.model import PolicyNetwork

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device(config_device):
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

def train_predictor(config, args):
    device = get_device(config['training']['device'])
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs/predictor", timestamp)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Load pre-trained encoder
    print(f"Loading pre-trained encoder from {args.encoder_path}...")
    policy_net = PolicyNetwork(hidden_dim=config['model']['hidden_dim'])
    
    # Load checkpoint
    checkpoint = torch.load(args.encoder_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        policy_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy_net.load_state_dict(checkpoint)
    
    encoder = policy_net.canvas_encoder
    encoder.to(device)
    encoder.eval()  # Freeze encoder
    
    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    print("Encoder loaded and frozen")
    
    # Get latent dimension from encoder
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    with torch.no_grad():
        encoder_out = encoder(dummy_input)
        latent_dim = encoder_out.view(encoder_out.size(0), -1).shape[1]  # Flatten and get dim
    print(f"Latent dimension: {latent_dim}")
    
    # Dataset
    print("Creating dataset...")
    categories = config['data']['categories']
    if categories == "all" or categories == -1:
        from picassbot.quickdraw import QuickDrawDataset
        qd = QuickDrawDataset(config['data']['data_dir'])
        categories = qd.categories
        print(f"Using ALL {len(categories)} categories")
    
    dataset = SimpleLatentDynamicsDataset(
        config['data']['data_dir'],
        categories,
        encoder,
        device,
        max_samples=config['data']['max_drawings_per_category']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)} transitions")
    
    # Model
    predictor_type = config.get('predictor', {}).get('type', 'lstm')
    hidden_dim = config.get('predictor', {}).get('hidden_dim', 512)
    num_layers = config.get('predictor', {}).get('num_layers', 2)
    
    if predictor_type == 'gru':
        predictor = LatentPredictorGRU(
            latent_dim=latent_dim,
            action_dim=4,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        print(f"Using GRU predictor (hidden_dim={hidden_dim}, layers={num_layers})")
    else:
        predictor = LatentPredictor(
            latent_dim=latent_dim,
            action_dim=4,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        print(f"Using LSTM predictor (hidden_dim={hidden_dim}, layers={num_layers})")
    
    # Optimizer
    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Loss
    criterion = nn.MSELoss()
    
    # Training loop
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        predictor.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, (z_t, a_t, z_target) in enumerate(pbar):
            z_t = z_t.to(device)
            a_t = a_t.to(device)
            z_target = z_target.to(device)
            
            # Forward
            z_pred, _ = predictor(z_t, a_t)
            
            # Loss
            loss = criterion(z_pred, z_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if global_step % config['training']['log_every'] == 0:
                writer.add_scalar('Loss/mse', loss.item(), global_step)
                writer.add_scalar('Learning/lr', optimizer.param_groups[0]['lr'], global_step)
            
            pbar.set_postfix({'loss': loss.item()})
            global_step += 1
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            os.makedirs("predictor_checkpoints", exist_ok=True)
            checkpoint_path = f"predictor_checkpoints/predictor_epoch_{epoch+1}.pth"
            
            # Save to temporary file first
            temp_path = checkpoint_path + ".tmp"
            try:
                # Try lightweight save first (model weights only)
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': predictor.state_dict(),
                    'loss': avg_loss,
                    'latent_dim': latent_dim
                }
                
                torch.save(checkpoint_data, temp_path)
                
                # If successful, rename to final path
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                os.rename(temp_path, checkpoint_path)
                
                print(f"Saved checkpoint: {checkpoint_path}")
                
                # Check file size
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"  Checkpoint size: {size_mb:.1f} MB")
                
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
                print(f"  This may be due to insufficient disk space (only 152 MB available)")
                print(f"  Consider freeing up disk space or using a different checkpoint directory")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
    
    writer.close()
    print(f"\nTraining complete! Logs: {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--encoder_path", type=str, required=True, 
                       help="Path to pre-trained policy network checkpoint")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Add predictor-specific config if not present
    if 'predictor' not in config:
        config['predictor'] = {
            'type': 'lstm',
            'hidden_dim': 512,
            'num_layers': 2
        }
    
    train_predictor(config, args)
