
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from skimage.metrics import structural_similarity as ssim
import csv

from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 10007
IMAGE_INDEX = 0

SIGMAS = [0.0, 0.05, 0.10]
NOISE_SAMPLES = 5
IG_STEPS = 20

OUTPUT_DIR = RESULTS_DIR / "official_protocol_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_A_NAME = f"resnet34_cifar_N8000_seed{SEED}_modelA"
MODEL_B_NAME = f"resnet34_cifar_N8000_seed{SEED}_modelB"

torch.manual_seed(SEED)
np.random.seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.set_device(0)
    print("GPU :", torch.cuda.get_device_name(0))
else:
    torch.set_num_threads(2)
    print("Utilisation CPU")

# ==================================================
# 2. MODELE
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


def load_model(name):
    path = CHECKPOINT_DIR / name / "final_model.pth"

    if not path.exists():
        raise FileNotFoundError(path)

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

# ==================================================
# 3. IMAGE
# ==================================================

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

print("Image :", IMAGE_INDEX)
print("Vraie classe :", true_label)

# ==================================================
# 4. GRADIENTS
# ==================================================

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


# ==================================================
# 5. CARTES MOYENNES AVEC BRUITS COMMUNS
# ==================================================

def averaged_map(model, method, noise_list):
    total = torch.zeros_like(x)

    for noise in noise_list:
        noisy_x = x + noise

        if method == "Simple-Grad":
            attribution = simple_grad(model, noisy_x, y)
        else:
            attribution = integrated_grad(model, noisy_x, y)

        total += attribution

    return total / len(noise_list)


def spatial_map(attribution):
    # Moyenne des valeurs absolues sur les canaux
    result = attribution.abs().sum(dim=1)[0]
    return result.detach().cpu().numpy().astype(np.float32)


def normalized_map(array):
    minimum = array.min()
    maximum = array.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(array)

    return (array - minimum) / (maximum - minimum)


# ==================================================
# 6. EXPERIMENTATION
# ==================================================

rows = []

for method in ["Simple-Grad", "Integrated-Grad"]:

    print("\n================================")
    print("METHODE :", method)
    print("================================")


    for sigma in SIGMAS:

        sigma_noise = 2.0 * sigma

        # Les mêmes bruits sont partagés par A et B
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED + int(sigma * 10000))

        noise_list = []

        for _ in range(NOISE_SAMPLES):
            noise_cpu = torch.randn(
                x.shape,
                generator=generator,
                dtype=x.dtype,
                device="cpu"
            )

            noise_list.append(
                (noise_cpu * sigma_noise).to(DEVICE)
            )

        with torch.enable_grad():
            map_a = averaged_map(model_a, method, noise_list)
            map_b = averaged_map(model_b, method, noise_list)

            print("Dimensions image :", x.shape)
            print("Dimensions carte A :", map_a.shape)
            print("Dimensions carte B :", map_b.shape)
            print("Dimensions après spatial_map :", spatial_map(map_a).shape)

        # L2 sur les gradients signés complets
        l2_signed = torch.linalg.vector_norm(
            (map_a - map_b).reshape(-1),
            ord=2
        ).item()


        # Transformation réservée au calcul du SSIM
        array_a = spatial_map(map_a)
        array_b = spatial_map(map_b)

        norm_a = normalized_map(array_a)
        norm_b = normalized_map(array_b)

        ssim_value = float(
            ssim(
                norm_a,
                norm_b,
                data_range=1.0
            )
        )

        rows.append({
            "seed": SEED,
            "image_index": IMAGE_INDEX,
            "method": method,
            "sigma_relative": sigma,
            "sigma_noise": sigma_noise,
            "noise_samples": NOISE_SAMPLES,
            "L2_signed_tensor_A_B": l2_signed,
            "SSIM_A_B": ssim_value
        })

        print(
            f"sigma={sigma:.2f} | "
            f"L2 signée A/B={l2_signed:.6f} | "
            f"SSIM A/B={ssim_value:.6f}"
        )

# ==================================================
# 7. SAUVEGARDE
# ==================================================

csv_path = OUTPUT_DIR / "one_image_stability_AB.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nRésultats sauvegardés :", csv_path)

del model_a, model_b

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print("\nTEST DE STABILITE A/B TERMINE")
