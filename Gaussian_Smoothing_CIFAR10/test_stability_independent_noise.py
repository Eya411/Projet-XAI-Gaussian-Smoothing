
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    DEVICE,
)

# Paramètres du test réduit
SEED = 10007
IMAGE_INDEX = 3535
SIGMA_RELATIVE = 0.05
NOISE_SAMPLES = 5
INPUT_RANGE = 2.0
IG_STEPS = 20

EXPLANATIONS_DIR = Path("results/phase6_final/explanations")


def create_model():
    model = models.resnet34(weights=None)

    # Adaptation ResNet34 à CIFAR-10 (32x32)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def load_model(seed, part):
    model = create_model()

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"resnet34_cifar_N8000_seed{seed}_model{part}"
        / "final_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = checkpoint

        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Modèle chargé : seed={seed}, part={part}")
    return model


def simple_gradient(model, inputs, label):
    x = inputs.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    output = model(x)
    score = output[:, label].sum()
    score.backward()

    return x.grad.detach()


def integrated_gradients(model, inputs, label):
    baseline = torch.zeros_like(inputs)
    total_gradient = torch.zeros_like(inputs)

    for alpha in np.linspace(0.0, 1.0, IG_STEPS):
        interpolated = (
            baseline + alpha * (inputs - baseline)
        ).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        output = model(interpolated)
        score = output[:, label].sum()
        score.backward()

        total_gradient += interpolated.grad.detach()

    return (
        (inputs - baseline)
        * total_gradient
        / IG_STEPS
    )


def averaged_noisy_explanation(model, image, label, method, sigma):
    total_map = torch.zeros_like(image)

    for _ in range(NOISE_SAMPLES):
        # Chaque appel génère un bruit indépendant
        noise = torch.randn_like(image) * sigma
        noisy_image = image + noise

        if method == "SimpleGrad":
            explanation = simple_gradient(
                model, noisy_image, label
            )
        else:
            explanation = integrated_gradients(
                model, noisy_image, label
            )

        total_map += explanation

    return total_map / NOISE_SAMPLES


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)

    print("Device :", DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZE_MEAN,
            NORMALIZE_STD,
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform,
    )

    image, label = test_dataset[IMAGE_INDEX]
    image = image.unsqueeze(0).to(DEVICE)

    print("Image :", IMAGE_INDEX)
    print("Vrai label :", label)

    sigma_noise = SIGMA_RELATIVE * INPUT_RANGE

    print("Sigma relatif :", SIGMA_RELATIVE)
    print("Sigma appliqué à l'entrée :", sigma_noise)
    print("Nombre de bruits :", NOISE_SAMPLES)

    model_a = load_model(SEED, "A")
    model_b = load_model(SEED, "B")

    methods = ["SimpleGrad", "IntegratedGradients"]

    for method in methods:
        print(f"\nMéthode : {method}")

        # Bruits indépendants pour A et B :
        # chaque fonction génère ses propres perturbations.
        map_a = averaged_noisy_explanation(
            model_a, image, label, method, sigma_noise
        )

        map_b = averaged_noisy_explanation(
            model_b, image, label, method, sigma_noise
        )

        distance = torch.norm(
            (map_a - map_b).reshape(-1), p=2
        ).item()

        print("Norme L2 entre les cartes moyennes :", distance)

        print("Carte A finie :", torch.isfinite(map_a).all().item())
        print("Carte B finie :", torch.isfinite(map_b).all().item())


if __name__ == "__main__":
    main()
