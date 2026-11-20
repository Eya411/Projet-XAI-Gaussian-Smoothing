
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.cuda.set_device(0)
    print("GPU utilise :", torch.cuda.get_device_name(0))
else:
    torch.set_num_threads(2)
    print("GPU indisponible : utilisation CPU")

MODEL_NAME = "resnet34_cifar_N8000_seed10007_modelA"

IMAGE_INDEX = 0

IG_STEPS = 20
SMOOTHGRAD_SAMPLES = 100
SMOOTHGRAD_NOISE_STD = 0.1

OUTPUT_DIR = RESULTS_DIR / "phase6_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ==========================================
# 2. CREATION DU MODELE
# ==========================================

def create_model():
    model = models.resnet34(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


# ==========================================
# 3. CHARGEMENT DU CHECKPOINT
# ==========================================

checkpoint_path = CHECKPOINT_DIR / MODEL_NAME / "final_model.pth"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint introuvable : {checkpoint_path}"
    )

model = create_model()

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

print("Informations du checkpoint :", list(checkpoint.keys()))

state_dict = checkpoint["model_state_dict"]

model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()

print("Modele charge avec succes :", MODEL_NAME)

print("Modele charge :", MODEL_NAME)


# ==========================================
# 4. CHARGEMENT D'UNE IMAGE
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )
])

dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

image, true_label = dataset[IMAGE_INDEX]

input_tensor = image.unsqueeze(0).to(DEVICE)
input_tensor.requires_grad_(True)

original_image = (
    dataset.data[IMAGE_INDEX].astype(np.float32) / 255.0
)

print("Image test :", IMAGE_INDEX)
print("Classe reelle :", CLASS_NAMES[true_label])


# ==========================================
# 5. PREDICTION
# ==========================================

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)

    predicted_class = int(logits.argmax(dim=1).item())
    confidence = float(
        probabilities[0, predicted_class].item()
    )

print("Classe predite :", CLASS_NAMES[predicted_class])
print(f"Confiance : {confidence * 100:.2f} %")

# On explique la classe prédite
target = predicted_class


# ==========================================
# 6. SIMPLE GRADIENT
# ==========================================

print("\nCalcul de Simple Gradient...")

saliency = Saliency(model)

simple_gradient = saliency.attribute(
    input_tensor,
    target=target,
    abs=True
)

simple_map = simple_gradient.detach().cpu().numpy()[0]

print("Simple Gradient termine.")


# ==========================================
# 7. INTEGRATED GRADIENTS
# ==========================================

print("\nCalcul de Integrated Gradients...")

integrated_gradients = IntegratedGradients(model)

baseline = torch.zeros_like(input_tensor)

ig_attributions, convergence_delta = integrated_gradients.attribute(
    input_tensor,
    baselines=baseline,
    target=target,
    n_steps=IG_STEPS,
    method="gausslegendre",
    return_convergence_delta=True
)

ig_map = ig_attributions.detach().cpu().numpy()[0]

print(
    "Integrated Gradients termine. "
    "Convergence delta :",
    float(convergence_delta.detach().cpu().item())
)


# ==========================================
# 8. SMOOTHGRAD
# ==========================================

print("\nCalcul de SmoothGrad...")

noise_tunnel = NoiseTunnel(saliency)

smoothgrad = noise_tunnel.attribute(
    input_tensor,
    nt_type="smoothgrad",
    nt_samples=SMOOTHGRAD_SAMPLES,
    nt_samples_batch_size=10,
    stdevs=SMOOTHGRAD_NOISE_STD,
    target=target,
    abs=True
)

smoothgrad_map = smoothgrad.detach().cpu().numpy()[0]

print("SmoothGrad termine.")


# ==========================================
# 9. PREPARATION DES CARTES
# ==========================================

def to_heatmap(attribution):
    heatmap = np.abs(attribution).sum(axis=0)

    maximum = heatmap.max()

    if maximum > 0:
        heatmap = heatmap / maximum

    return heatmap.astype(np.float32)


maps = {
    "Simple Gradient": to_heatmap(simple_map),
    "Integrated Gradients": to_heatmap(ig_map),
    "SmoothGrad": to_heatmap(smoothgrad_map)
}


# ==========================================
# 10. SAUVEGARDE DES RESULTATS
# ==========================================

np.savez_compressed(
    OUTPUT_DIR / "xai_one_image_maps.npz",
    original_image=original_image,
    simple_gradient=maps["Simple Gradient"],
    integrated_gradients=maps["Integrated Gradients"],
    smoothgrad=maps["SmoothGrad"],
    true_label=true_label,
    predicted_class=predicted_class,
    confidence=confidence
)

print("Cartes XAI sauvegardees.")


# ==========================================
# 11. VISUALISATION
# ==========================================

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].imshow(original_image)
axes[0].set_title("Image originale")

for ax, (name, heatmap) in zip(axes[1:], maps.items()):
    ax.imshow(original_image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.set_title(name)

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"Classe reelle : {CLASS_NAMES[true_label]} | "
    f"Prediction : {CLASS_NAMES[predicted_class]}"
)

plt.tight_layout()

figure_path = OUTPUT_DIR / "xai_one_image_comparison.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close()

print("\n========================================")
print("TEST XAI TERMINE")
print("Resultats :", OUTPUT_DIR)
print("Image :", figure_path)
print("========================================")

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
