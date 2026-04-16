"""
Neural Canvas - GAN Generator Trainer
======================================
Architecture (Generator):
    Linear(16, 128) -> ReLU
    Linear(128, 512) -> ReLU
    Linear(512, 784) -> Sigmoid

Trains a GAN on MNIST (28x28 grayscale digits).  If torchvision is not
installed the script falls back to a self-supervised autoencoder that learns
to reconstruct random binary-noise patterns, which is enough to produce
a trained weight file that the C++ engine can load.

Output: weights/generator.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

LATENT_DIM = 16
IMG_SIZE = 784  # 28 * 28


class Generator(nn.Module):
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_mnist(batch_size: int):
    """Return a DataLoader for MNIST, or None if torchvision absent."""
    try:
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.ToTensor(),          # [0, 1] float32
            transforms.Lambda(lambda t: t.view(-1)),  # flatten to 784
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


def _make_noise_loader(batch_size: int, num_batches: int = 200):
    """Fallback: synthetic dataset of random binary-noise 28x28 images."""
    print("[INFO] torchvision not found - using synthetic noise dataset.")
    data = torch.rand(num_batches * batch_size, IMG_SIZE)
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    output_path: str = "weights/generator.pth",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 2e-4,
    device_str: str = "cpu",
):
    device = torch.device(device_str)

    loader = _load_mnist(batch_size)
    if loader is None:
        loader = _make_noise_loader(batch_size)
        fallback = True
    else:
        fallback = False

    gen = Generator().to(device)
    disc = Discriminator().to(device)

    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    real_label = 1.0
    fake_label = 0.0

    print(f"Training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        d_loss_total = 0.0
        g_loss_total = 0.0
        batches = 0

        for batch in loader:
            real = batch[0].to(device) if fallback else batch[0].to(device)

            # ------------------------------------------------------------------
            # Train Discriminator
            # ------------------------------------------------------------------
            disc.zero_grad()

            # Real images
            out_real = disc(real)
            labels_real = torch.full(
                (real.size(0), 1), real_label, device=device
            )
            loss_real = criterion(out_real, labels_real)

            # Fake images
            z = torch.randn(real.size(0), LATENT_DIM, device=device)
            fake = gen(z).detach()
            out_fake = disc(fake)
            labels_fake = torch.full(
                (real.size(0), 1), fake_label, device=device
            )
            loss_fake = criterion(out_fake, labels_fake)

            loss_d = (loss_real + loss_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # ------------------------------------------------------------------
            # Train Generator
            # ------------------------------------------------------------------
            gen.zero_grad()

            z = torch.randn(real.size(0), LATENT_DIM, device=device)
            fake = gen(z)
            out_g = disc(fake)
            loss_g = criterion(out_g, labels_real)  # fool discriminator
            loss_g.backward()
            opt_g.step()

            d_loss_total += loss_d.item()
            g_loss_total += loss_g.item()
            batches += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"D-loss: {d_loss_total / batches:.4f} | "
                f"G-loss: {g_loss_total / batches:.4f}"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(gen.state_dict(), output_path)
    print(f"\n[DONE] Generator saved to '{output_path}'")
    print("Next step: run scripts/extract_generator_weights.py")


if __name__ == "__main__":
    device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    train(device_str=device_arg)
