
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES
)

# ==================================================
# 1. CONFIGURATION DU TEST
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "resnet34_cifar_N8000_seed10007_modelA"
IMAGE_INDEX = 0

# Quelques valeurs pour valider le fonctionnement
SIGMAS_RELATIVE = [0.0, 0.05, 0.10]

# Test réduit pour limiter la charge GPU
NOISE_SAMPLES = 5
IG_STEPS = 20

OUTPUT_DIR = RESULTS_DIR / "official_protocol_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if DEVICE.type == "cuda":
    print("GPU :", torch.cuda.get_device_name(0))
else:
    torch.set_num_threads(2)
    print("Utilisation CPU")

# Reproductibilité
torch.manual_seed(10007)
np.random.seed(10007)

# ==================================================
# 2. CREATION DU MODELE
# ==================================================

def create_model():
    model = models.resnet34(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3,
        stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


# ==================================================
# 3. CHARGEMENT DU CHECKPOINT
# ==================================================

checkpoint_path = (
    CHECKPOINT_DIR / MODEL_NAME / "final_model.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model = create_model()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Modèle chargé :", MODEL_NAME)

# ==================================================
# 4. CHARGEMENT DE L'IMAGE
# ==================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD
    )
])

dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

image, true_label = dataset[IMAGE_INDEX]
images = image.unsqueeze(0).to(DEVICE)
labels = torch.tensor([true_label], device=DEVICE)

original_image = (
    dataset.data[IMAGE_INDEX].astype(np.float32) / 255.0
)

print("Image :", IMAGE_INDEX)
print("Vraie classe :", true_label)

# ==================================================
# 5. PREDICTION
# ==================================================

with torch.no_grad():
    logits = model(images)
    predicted_class = int(logits.argmax(dim=1).item())

print("Classe prédite :", predicted_class)

# Le protocole officiel explique le score de la vraie classe.
target = labels

# ==================================================
# 6. SIMPLE GRADIENT SIGNE
# ==================================================

def calc_simple_grad(net, x, y):
    x_grad = x.detach().clone().requires_grad_(True)

    logits = net(x_grad)
    score = logits.gather(1, y.view(-1, 1)).sum()

    net.zero_grad(set_to_none=True)
    score.backward()

    return x_grad.grad.detach()


# ==================================================
# 7. INTEGRATED GRADIENT - 20 POINTS
# ==================================================

def calc_integrated_grad(net, x, y, steps=20):
    baseline = torch.zeros_like(x)
    total_gradient = torch.zeros_like(x)

    for alpha in np.linspace(0, 1, steps):
        interpolated = (
            baseline + alpha * (x - baseline)
        ).detach().requires_grad_(True)

        logits = net(interpolated)
        score = logits.gather(1, y.view(-1, 1)).sum()

        net.zero_grad(set_to_none=True)
        score.backward()

        total_gradient += interpolated.grad.detach()

    return (x - baseline) * total_gradient / steps


# ==================================================
# 8. BRUIT GAUSSIEN ET LISSAGE
# ==================================================

def smoothed_explanation(
    net, x, y, sigma_relative,
    method="simple", samples=5
):
    # Normalisation CIFAR-10 : intervalle théorique [-1,1]
    sigma_noise = sigma_relative * 2.0

    accumulated = torch.zeros_like(x)

    for _ in range(samples):
        noise = torch.randn_like(x) * sigma_noise
        noisy_input = x + noise

        if method == "simple":
            attribution = calc_simple_grad(
                net, noisy_input, y
            )
        elif method == "integrated":
            attribution = calc_integrated_grad(
                net, noisy_input, y, IG_STEPS
            )
        else:
            raise ValueError("Méthode inconnue")

        accumulated += attribution

    return accumulated / samples


# ==================================================
# 9. TEST FIDELITE
# ==================================================

results = []

methods = {
    "Simple-Grad": "simple",
    "Integrated-Grad": "integrated"
}

with torch.enable_grad():
    for method_name, method_code in methods.items():

        print("\nMéthode :", method_name)

        if method_code == "simple":
            reference = calc_simple_grad(model, images, target)
        else:
            reference = calc_integrated_grad(
                model, images, target, IG_STEPS
            )

        for sigma in SIGMAS_RELATIVE:

            smoothed = smoothed_explanation(
                model,
                images,
                target,
                sigma,
                method=method_code,
                samples=NOISE_SAMPLES
            )

            # Distance L2 comme dans le dépôt officiel
            fidelity = torch.norm(
                (smoothed - reference).reshape(1, -1),
                dim=1
            ).mean().item()

            results.append({
                "method": method_name,
                "sigma_relative": sigma,
                "sigma_noise": 2.0 * sigma,
                "fidelity_L2": fidelity
            })

            print(
                f"sigma={sigma:.2f} | "
                f"bruit={2*sigma:.2f} | "
                f"L2={fidelity:.6f}"
            )

# ==================================================
# 10. SAUVEGARDE
# ==================================================

import csv

csv_path = OUTPUT_DIR / "one_image_fidelity.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method", "sigma_relative",
            "sigma_noise", "fidelity_L2"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nRésultats sauvegardés :", csv_path)

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print("\nTEST TERMINE")
