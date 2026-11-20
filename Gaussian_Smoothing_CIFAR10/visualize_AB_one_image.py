
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models

from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 10007
IMAGE_INDEX = 0
IG_STEPS = 20

OUTPUT_DIR = RESULTS_DIR / "official_protocol_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_A_NAME = f"resnet34_cifar_N8000_seed{SEED}_modelA"
MODEL_B_NAME = f"resnet34_cifar_N8000_seed{SEED}_modelB"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

if DEVICE.type == "cuda":
    torch.cuda.set_device(0)
    print("GPU :", torch.cuda.get_device_name(0))
else:
    torch.set_num_threads(2)
    print("Utilisation CPU")

# ==========================================
# 2. MODELE
# ==========================================

def create_model():
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3,
        stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model(name):
    path = CHECKPOINT_DIR / name / "final_model.pth"

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Modèle chargé :", name)
    return model


model_a = load_model(MODEL_A_NAME)
model_b = load_model(MODEL_B_NAME)

# ==========================================
# 3. IMAGE
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

image, true_label = dataset[IMAGE_INDEX]
x = image.unsqueeze(0).to(DEVICE)
y = torch.tensor([true_label], device=DEVICE)

original_image = (
    dataset.data[IMAGE_INDEX].astype(np.float32) / 255.0
)

print("Image :", IMAGE_INDEX)
print("Vraie classe :", CLASS_NAMES[true_label])

# ==========================================
# 4. EXPLICATIONS SIGNEES
# ==========================================

def simple_grad(model, inputs, labels):
    inputs = inputs.detach().clone().requires_grad_(True)

    logits = model(inputs)
    score = logits.gather(1, labels.view(-1, 1)).sum()

    model.zero_grad(set_to_none=True)
    score.backward()

    return inputs.grad.detach()


def integrated_grad(model, inputs, labels):
    baseline = torch.zeros_like(inputs)
    total_gradient = torch.zeros_like(inputs)

    for alpha in np.linspace(0, 1, IG_STEPS):
        point = (
            baseline + alpha * (inputs - baseline)
        ).detach().requires_grad_(True)

        logits = model(point)
        score = logits.gather(1, labels.view(-1, 1)).sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        total_gradient += point.grad.detach()

    return (inputs - baseline) * total_gradient / IG_STEPS


# ==========================================
# 5. CALCUL A ET B, SIGMA = 0
# ==========================================

with torch.enable_grad():
    simple_a = simple_grad(model_a, x, y)
    simple_b = simple_grad(model_b, x, y)

    ig_a = integrated_grad(model_a, x, y)
    ig_b = integrated_grad(model_b, x, y)

print("Simple-Grad A :", tuple(simple_a.shape))
print("Simple-Grad B :", tuple(simple_b.shape))
print("Integrated-Grad A :", tuple(ig_a.shape))
print("Integrated-Grad B :", tuple(ig_b.shape))


# ==========================================
# 6. TRANSFORMATION POUR AFFICHAGE
# ==========================================

def display_map(attribution):
    # Transformation visuelle uniquement.
    # Les tenseurs originaux signés restent inchangés.
    result = attribution.abs().sum(dim=1)[0]
    result = result.detach().cpu().numpy()

    maximum = result.max()
    if maximum > 0:
        result = result / maximum

    return result


maps = [
    ("Simple-Grad A", display_map(simple_a)),
    ("Simple-Grad B", display_map(simple_b)),
    ("Integrated-Grad A", display_map(ig_a)),
    ("Integrated-Grad B", display_map(ig_b))
]

# ==========================================
# 7. SAUVEGARDE DES CARTES
# ==========================================

np.savez_compressed(
    OUTPUT_DIR / "AB_signed_maps_sigma0.npz",
    simple_a=simple_a.detach().cpu().numpy(),
    simple_b=simple_b.detach().cpu().numpy(),
    integrated_a=ig_a.detach().cpu().numpy(),
    integrated_b=ig_b.detach().cpu().numpy(),
    true_label=true_label,
    image_index=IMAGE_INDEX,
    sigma=0.0
)

# ==========================================
# 8. FIGURE
# ==========================================

fig, axes = plt.subplots(1, 5, figsize=(17, 4))

axes[0].imshow(original_image)
axes[0].set_title(
    f"Image originale\n{CLASS_NAMES[true_label]}"
)

for ax, (name, heatmap) in zip(axes[1:], maps):
    ax.imshow(original_image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.set_title(name)

for ax in axes:
    ax.axis("off")

fig.suptitle(
    "Comparaison A/B — Sigma = 0 — Image 0",
    fontsize=13
)

plt.tight_layout()

figure_path = OUTPUT_DIR / "AB_visual_comparison_sigma0.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close()

print("\nCartes sauvegardées.")
print("Figure :", figure_path)
print("Fichier NPZ :", OUTPUT_DIR / "AB_signed_maps_sigma0.npz")

del model_a, model_b

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print("\nVISUALISATION TERMINEE")
