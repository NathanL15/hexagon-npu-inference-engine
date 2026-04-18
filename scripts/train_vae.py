"""train a vae and save only decoder weights to weights/generator.pth."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


LATENT_DIM = 16
IMG_SIZE    = 784   # 28 * 28
BETA        = 1.0


class Encoder(nn.Module):
    """Maps a flat 784-dim image to (mu, log_var) in R^LATENT_DIM."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(IMG_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(128, LATENT_DIM)
        self.fc_log_var = nn.Linear(128, LATENT_DIM)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """same layout as the gan generator so extractor keys still match."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, IMG_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        """reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var


def vae_loss(recon, target, mu, log_var, beta: float = BETA):
    """elbo with bce recon term and gaussian kl term."""
    recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / target.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def _load_mnist(batch_size: int):
    """Return a DataLoader for MNIST, or None if torchvision absent."""
    try:
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.view(-1)),
        ])
        dataset = torchvision.datasets.MNIST(
            root="weights/data",
            train=True,
            download=True,
            transform=transform,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        print("[INFO] Using MNIST dataset.")
        return loader
    except ImportError:
        return None


def _make_noise_loader(batch_size: int, num_batches: int = 400):
    """fallback random patterns if torchvision is unavailable."""
    print("[INFO] torchvision not found -- using synthetic noise dataset.")
    data    = torch.rand(num_batches * batch_size, IMG_SIZE)
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(
    output_path: str = "weights/generator.pth",
    epochs:      int   = 50,
    batch_size:  int   = 128,
    lr:          float = 1e-3,
    device_str:  str   = "cpu",
):
    device = torch.device(device_str)

    loader = _load_mnist(batch_size)
    if loader is None:
        loader = _make_noise_loader(batch_size)

    model     = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training VAE for {epochs} epochs on {device}  "
          f"(LATENT_DIM={LATENT_DIM}, beta={BETA})...")

    for epoch in range(1, epochs + 1):
        total_loss   = 0.0
        total_recon  = 0.0
        total_kl     = 0.0
        batches      = 0

        for batch in loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            recon, mu, log_var = model(x)
            loss, recon_l, kl_l = vae_loss(recon, x, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_l.item()
            total_kl    += kl_l.item()
            batches     += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss: {total_loss / batches:8.2f} | "
                f"recon: {total_recon / batches:8.2f} | "
                f"KL: {total_kl / batches:6.2f}"
            )

    # save only the decoder so extractor and c++ runtime stay compatible.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.decoder.state_dict(), output_path)
    print(f"\n[DONE] Decoder saved to '{output_path}'")
    print("The latent space is now a smooth Gaussian ball -- no more static.")
    print("Next step: run scripts/extract_generator_weights.py")


if __name__ == "__main__":
    device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    train(device_str=device_arg)
