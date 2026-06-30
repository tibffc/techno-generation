import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.diversity.extractor_universal import AudioFeatureExtractor


class VAEDataset(Dataset):
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]


class VAE(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


class VAETrainer:
    def __init__(self, model, device='cuda', learning_rate=1e-3, 
                 patience=10, min_delta=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def early_stop_check(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                print(f"\n Early stopping after {self.epochs_without_improvement} epochs without improvement")
                return True
            return False
    
    def train_epoch(self, dataloader, beta=1.0):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            recon, mu, log_var = self.model(batch)
            loss, recon_loss, kl_loss = self.loss_function(recon, batch, mu, log_var, beta)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        n = len(dataloader)
        return total_loss / n, total_recon / n, total_kl / n
    
    def validate(self, dataloader, beta=1.0):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                recon, mu, log_var = self.model(batch)
                loss, _, _ = self.loss_function(recon, batch, mu, log_var, beta)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def restore_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("   Restored best model from early stopping checkpoint")


def extract_embeddings(audio_dir, backend='wav2vec2_music', duration=10.0):
    """Extract embeddings from audio directory."""
    print(f"\n Extracting embeddings from {audio_dir}...")
    extractor = AudioFeatureExtractor(backend=backend, duration=duration)
    embeddings, paths = extractor.extract_from_directory(audio_dir)
    print(f"   Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    return embeddings, paths


def load_embeddings(embeddings_path):
    """Load pre-extracted embeddings from .npy file."""
    print(f"\n Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"   Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Train VAE on audio embeddings")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio_dir', help='Directory with audio files (will extract embeddings)')
    input_group.add_argument('--embeddings_path', help='Path to pre-extracted embeddings .npy file')
    
    parser.add_argument('--output_dir', required=True, help='Output directory for model')
    parser.add_argument('--backend', default='wav2vec2_music', 
                        choices=['librosa', 'wav2vec2', 'wav2vec2_music'])
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--beta', type=float, default=1.0, help='KL divergence weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Min improvement for early stopping')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration for audio extraction')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f" VAE TRAINING (with Early Stopping)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Early stopping patience: {args.patience}")
    
    # Load or extract embeddings
    if args.embeddings_path:
        embeddings = load_embeddings(args.embeddings_path)
    else:
        embeddings, _ = extract_embeddings(args.audio_dir, args.backend, args.duration)
    
    # Split train/val
    n = len(embeddings)
    n_train = int(0.8 * n)
    indices = np.random.permutation(n)
    train_emb = embeddings[indices[:n_train]]
    val_emb = embeddings[indices[n_train:]]
    
    print(f"\n Dataset: {len(train_emb)} train, {len(val_emb)} val")
    
    # Dataloaders
    train_loader = DataLoader(VAEDataset(train_emb), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VAEDataset(val_emb), batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = VAE(input_dim=embeddings.shape[1], latent_dim=args.latent_dim)
    print(f" Model: {embeddings.shape[1]} → {args.latent_dim} → {embeddings.shape[1]}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = VAETrainer(model, device, args.lr, args.patience, args.min_delta)
    
    # Training
    print(f"\n Training (max {args.epochs} epochs)...")
    history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(args.epochs):
        train_loss, recon_loss, kl_loss = trainer.train_epoch(train_loader, args.beta)
        val_loss = trainer.validate(val_loader, args.beta)
        
        trainer.scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(recon_loss)
        history['kl_loss'].append(kl_loss)
        
        # Early stopping check
        if trainer.early_stop_check(val_loss):
            print(f"\n Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Recon: {recon_loss:.6f} | KL: {kl_loss:.6f}")
    
    # Restore best model
    trainer.restore_best_model()
    
    # Save final model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'latent_dim': args.latent_dim,
        'input_dim': embeddings.shape[1],
        'train_loss': train_loss,
        'val_loss': trainer.best_val_loss,
        'epochs_completed': epoch + 1,
    }, output_dir / 'best_model.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n Training complete!")
    print(f"   Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"   Model saved to: {output_dir}")


if __name__ == "__main__":
    main()