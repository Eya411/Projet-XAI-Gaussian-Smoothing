
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    DEVICE,
)

from xai_config import (
    SEEDS,
    TEST_IMAGE_INDICES,
    STABILITY_SIGMAS,
    NOISE_SAMPLES,
    IG_STEPS,
    INPUT_RANGE,
)

# =====================================================
# MODE DE TEST
# =====================================================

# True : petit test de vérification
# False : expérience complète
QUICK_TEST = False

if QUICK_TEST:
    selected_indices = TEST_IMAGE_INDICES[:2]
    selected_seeds = [10007]
    selected_sigmas = [0.05]
    noise_samples = 2
else:
    selected_indices = TEST_IMAGE_INDICES
    selected_seeds = SEEDS
    selected_sigmas = STABILITY_SIGMAS
    noise_samples = NOISE_SAMPLES

BATCH_SIZE = 16
OUTPUT_DIR = Path("results/phase6_final/stability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# ARCHITECTURE ET CHARGEMENT
# =====================================================

def create_model():
    model = models.resnet34(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1,
        padding=1, bias=False
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

    state = checkpoint

    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Modèle chargé : seed={seed}, part={part}")
    return model


# =====================================================
# MÉTHODES D'EXPLICATION
# =====================================================

def simple_gradient(model, images, labels):
    inputs = images.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(inputs)

    scores = outputs.gather(
        1, labels.view(-1, 1)
    ).sum()

    scores.backward()

    return inputs.grad.detach()


def integrated_gradients(model, images, labels):
    baseline = torch.zeros_like(images)
    total_gradient = torch.zeros_like(images)

    for alpha in np.linspace(0.0, 1.0, IG_STEPS):
        interpolated = (
            baseline + alpha * (images - baseline)
        ).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(interpolated)

        scores = outputs.gather(
            1, labels.view(-1, 1)
        ).sum()

        scores.backward()
        total_gradient += interpolated.grad.detach()

    return (
        (images - baseline)
        * total_gradient
        / IG_STEPS
    )


# =====================================================
# MOYENNE DES EXPLICATIONS SOUS BRUIT
# =====================================================

def averaged_noisy_explanation(
    model,
    images,
    labels,
    method,
    sigma,
    generator,
):
    total_map = torch.zeros_like(images)

    for _ in range(noise_samples):
        noise = torch.randn(
            images.shape,
            device=DEVICE,
            dtype=images.dtype,
            generator=generator,
        ) * sigma

        noisy_images = images + noise

        if method == "SimpleGrad":
            explanation = simple_gradient(
                model, noisy_images, labels
            )
        else:
            explanation = integrated_gradients(
                model, noisy_images, labels
            )

        total_map += explanation

    return total_map / noise_samples


# =====================================================
# EXPÉRIENCE DE STABILITÉ
# =====================================================

def main():
    print("Device :", DEVICE)
    print("Mode test réduit :", QUICK_TEST)
    print("Images :", len(selected_indices))
    print("Seeds :", selected_seeds)
    print("Sigmas :", selected_sigmas)
    print("Bruits :", noise_samples)

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

    subset = Subset(test_dataset, selected_indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    detailed_rows = []

    methods = ["SimpleGrad", "IntegratedGradients"]

    for seed in selected_seeds:
        print(f"\n========== SEED {seed} ==========")

        model_a = load_model(seed, "A")
        model_b = load_model(seed, "B")

        for sigma_index, sigma_relative in enumerate(selected_sigmas):
            sigma_input = sigma_relative * INPUT_RANGE

            print(
                f"\nSigma relatif={sigma_relative:.3f}, "
                f"sigma entrée={sigma_input:.3f}"
            )

            for method_index, method in enumerate(methods):
                print("Méthode :", method)

                # Générateurs séparés pour A et B.
                # Les perturbations sont donc indépendantes.
                generator_a = torch.Generator(
                    device=DEVICE
                )
                generator_b = torch.Generator(
                    device=DEVICE
                )

                base_seed = (
                    20260000
                    + seed * 100
                    + sigma_index * 10
                    + method_index * 2
                )

                generator_a.manual_seed(base_seed)
                generator_b.manual_seed(base_seed + 1)

                for batch_number, (images, labels) in enumerate(loader):
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    map_a = averaged_noisy_explanation(
                        model_a,
                        images,
                        labels,
                        method,
                        sigma_input,
                        generator_a,
                    )

                    map_b = averaged_noisy_explanation(
                        model_b,
                        images,
                        labels,
                        method,
                        sigma_input,
                        generator_b,
                    )

                    # Distance L2 entre les cartes moyennes
                    distances = torch.norm(
                        (map_a - map_b).flatten(start_dim=1),
                        p=2,
                        dim=1,
                    )

                    distances = distances.cpu().numpy()

                    start = batch_number * BATCH_SIZE

                    for j, distance in enumerate(distances):
                        image_index = selected_indices[start + j]

                        detailed_rows.append({
                            "seed": seed,
                            "method": method,
                            "sigma_relative": sigma_relative,
                            "sigma_input": sigma_input,
                            "image_index": image_index,
                            "distance_l2": float(distance),
                        })

                print(
                    "Images calculées pour cette configuration :",
                    len(selected_indices),
                )

        del model_a, model_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =================================================
    # SAUVEGARDE DES RÉSULTATS DÉTAILLÉS
    # =================================================

    detailed_path = OUTPUT_DIR / "stability_per_image.csv"

    with open(detailed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "method",
                "sigma_relative",
                "sigma_input",
                "image_index",
                "distance_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    print("\nFichier détaillé sauvegardé :", detailed_path)

    # =================================================
    # MOYENNE DES 32 IMAGES POUR CHAQUE SEED
    # =================================================

    per_seed = {}

    for row in detailed_rows:
        key = (
            row["seed"],
            row["method"],
            row["sigma_relative"],
        )

        per_seed.setdefault(key, []).append(row["distance_l2"])

    seed_rows = []

    for (seed, method, sigma), values in per_seed.items():
        seed_rows.append({
            "seed": seed,
            "method": method,
            "sigma_relative": sigma,
            "mean_distance_l2": float(np.mean(values)),
            "number_of_images": len(values),
        })

    seed_path = OUTPUT_DIR / "stability_per_seed.csv"

    with open(seed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "method",
                "sigma_relative",
                "mean_distance_l2",
                "number_of_images",
            ],
        )
        writer.writeheader()
        writer.writerows(seed_rows)

    print("Fichier par seed sauvegardé :", seed_path)

    # =================================================
    # MOYENNE ET ÉCART-TYPE ENTRE LES SEEDS
    # =================================================

    overall = {}

    for row in seed_rows:
        key = (row["method"], row["sigma_relative"])
        overall.setdefault(key, []).append(row["mean_distance_l2"])

    overall_rows = []

    for (method, sigma), values in overall.items():
        overall_rows.append({
            "method": method,
            "sigma_relative": sigma,
            "mean_across_seeds": float(np.mean(values)),
            "std_across_seeds": (
                float(np.std(values, ddof=1))
                if len(values) > 1 else 0.0
            ),
            "number_of_seeds": len(values),
        })

    overall_path = OUTPUT_DIR / "stability_summary.csv"

    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "sigma_relative",
                "mean_across_seeds",
                "std_across_seeds",
                "number_of_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(overall_rows)

    print("Résumé sauvegardé :", overall_path)
    print("\nExpérience terminée.")


if __name__ == "__main__":
    main()
