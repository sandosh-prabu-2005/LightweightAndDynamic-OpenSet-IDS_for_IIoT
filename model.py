"""
Model definitions for VAE-based Intrusion Detection System (IDS).

This module contains:
- Encoder: Maps input features to latent space (μ, log σ²)
- Decoder: Reconstructs features from latent codes
- TeacherClassifier: Multi-layer classifier for known classes
- VAEWithTeacher: Full VAE + Teacher model
- StudentNet: Student network for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder: input → 64 → 32 → latent_dim (μ, log σ²)
    """
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = torch.clamp(self.fc_logvar(h2), min=-10, max=10)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: latent → 32 → 64 → output_dim
    """
    def __init__(self, latent_dim=32, output_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)


class TeacherClassifier(nn.Module):
    """
    Teacher Classifier: latent → 128×4 → Softmax
    """
    def __init__(self, latent_dim=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, z):
        return F.softmax(self.net(z), dim=-1)


class VAEWithTeacher(nn.Module):
    """
    Full VAE + Teacher model combining:
    - Encoder: input → latent (μ, log σ²)
    - Decoder: latent → reconstructed input
    - TeacherClassifier: latent → class logits
    """
    def __init__(self, input_dim, latent_dim=32, n_classes=2):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = TeacherClassifier(latent_dim, n_classes)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        logits = self.classifier(mu)  # use mean for stable classification
        return recon, mu, logvar, logits

    @torch.no_grad()
    def reconstruction_error(self, x, device='cpu'):
        """Per-sample MSE reconstruction error (used for Stage 2 EVT)."""
        self.eval()
        x_t = torch.as_tensor(x, dtype=torch.float32).to(device)
        recon, _, _, _ = self(x_t)
        return F.mse_loss(recon, x_t, reduction='none').mean(dim=1).cpu().numpy()


class StudentNet(nn.Module):
    """
    Student Network for knowledge distillation.
    Smaller than teacher for efficient deployment.
    latent → 64 → n_classes
    """
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, z):
        return self.net(z)


def total_vae_loss(recon, x, mu, logvar, logits, y_labels, beta_kl=1.0):
    """
    Compute total VAE loss: L = Lr + beta_kl * LKL + Lc
    
    Args:
        recon: Reconstructed output from decoder
        x: Original input
        mu: Mean from encoder
        logvar: Log variance from encoder
        logits: Classification logits from teacher
        y_labels: Target class labels
        beta_kl: Weight for KL divergence (β-VAE parameter)
    
    Returns:
        (total_loss, Lr, LKL, Lc)
    """
    Lr = F.mse_loss(recon, x, reduction='mean')
    LKL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    Lc = F.cross_entropy(logits, y_labels)
    return Lr + beta_kl * LKL + Lc, Lr.item(), LKL.item(), Lc.item()
