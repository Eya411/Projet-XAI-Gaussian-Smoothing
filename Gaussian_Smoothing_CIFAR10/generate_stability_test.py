
from pathlib import Path
import csv
import json
import os
import math

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

from xai_config import (
    SEEDS,
    TEST_IMAGE_INDICES,
    IG_STEPS,
    INPUT_RANGE,
)


# =====================================================
# 1. CONFIGURATION DU TEST MINIMAL
# =====================================================

N_IMAGES = 1
NOISE_SAMPLES_SAFE = 2
BATCH_SIZE = 1

SELECTED_INDICES = [
    int(TEST_IMAGE_INDICES[0])
]

SELECTED_SEEDS = [10007]
SELECTED_SIGMAS = [0.05]

METHODS = [
    "SimpleGrad",
    "IntegratedGradients",
]

OUTPUT_DIR = Path(
    "results/phase6_final/stability_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAILED_PATH = OUTPUT_DIR / "stability_per_image.csv"
SEED_PATH = OUTPUT_DIR / "stability_per_seed.csv"
SUMMARY_PATH = OUTPUT_DIR / "stability_summary.csv"
METADATA_PATH = OUTPUT_DIR / "experiment_config.json"

FIELDNAMES = [
    "seed",
    "method",
    "sigma_relative",
    "sigma_input",
    "image_index",
    "distance_l2",
]


# =====================================================
# 2. VERIFICATION DE LA CONFIGURATION
# =====================================================

if len(SELECTED_INDICES) != N_IMAGES:
    raise ValueError("Nombre d'images incorrect.")

if len(set(SELECTED_INDICES)) != N_IMAGES:
    raise ValueError("Les indices doivent être uniques.")

if not set(SELECTED_SEEDS).issubset(set(SEEDS)):
    raise ValueError("Le seed sélectionné n'existe pas dans xai_config.py.")

if NOISE_SAMPLES_SAFE <= 0:
    raise ValueError("Le nombre de perturbations doit être positif.")

if BATCH_SIZE != 1:
    raise ValueError("Le batch size doit rester égal à 1.")

if IG_STEPS <= 0:
    raise ValueError("IG_STEPS doit être positif.")

if not torch.cuda.is_available() and str(DEVICE).startswith("cuda"):
    raise RuntimeError("CUDA est demandé mais n'est pas disponible.")

print("Configuration du test vérifiée.")


# =====================================================
# 3. ARCHITECTURE RESNET34
# =====================================================

def create_model():

    model = models.resnet34(weights=None)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model


def load_model(seed, part):

    model = create_model()

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"resnet34_cifar_N8000_seed{seed}_model{part}"
        / "final_model.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    state = checkpoint

    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):

            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):
                state = checkpoint[key]
                break

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    print(f"Modèle chargé : seed={seed}, modèle={part}")

    return model


# =====================================================
# 4. METHODES D'EXPLICATION
# =====================================================

def simple_gradient(model, images, labels):

    inputs = (
        images.detach()
        .clone()
        .requires_grad_(True)
    )

    model.zero_grad(set_to_none=True)

    outputs = model(inputs)

    scores = outputs.gather(
        1,
        labels.view(-1, 1),
    ).sum()

    scores.backward()

    return inputs.grad.detach()


def integrated_gradients(model, images, labels):

    baseline = torch.zeros_like(images)

    total_gradient = torch.zeros_like(images)

    alphas = np.linspace(
        0.0,
        1.0,
        IG_STEPS,
    )

    for alpha in alphas:

        interpolated = (
            baseline + float(alpha) * (images - baseline)
        ).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        outputs = model(interpolated)

        scores = outputs.gather(
            1,
            labels.view(-1, 1),
        ).sum()

        scores.backward()

        total_gradient += interpolated.grad.detach()

    attribution = (
        (images - baseline)
        * total_gradient
        / IG_STEPS
    )

    return attribution.detach()


# =====================================================
# 5. MOYENNE DES EXPLICATIONS SOUS BRUIT GAUSSIEN
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

    for _ in range(NOISE_SAMPLES_SAFE):

        noise = torch.randn(
            images.shape,
            device=images.device,
            dtype=images.dtype,
            generator=generator,
        ) * sigma

        noisy_images = images + noise

        if method == "SimpleGrad":

            explanation = simple_gradient(
                model,
                noisy_images,
                labels,
            )

        elif method == "IntegratedGradients":

            explanation = integrated_gradients(
                model,
                noisy_images,
                labels,
            )

        else:

            raise ValueError(
                f"Méthode inconnue : {method}"
            )

        total_map += explanation

        del noise, noisy_images, explanation

    return total_map / NOISE_SAMPLES_SAFE


# =====================================================
# 6. METADONNEES
# =====================================================

def save_metadata():

    metadata = {
        "experiment_type": "minimal_test",
        "images": SELECTED_INDICES,
        "seeds": SELECTED_SEEDS,
        "sigmas": SELECTED_SIGMAS,
        "methods": METHODS,
        "noise_samples": NOISE_SAMPLES_SAFE,
        "batch_size": BATCH_SIZE,
        "ig_steps": int(IG_STEPS),
        "input_range": float(INPUT_RANGE),
        "device": str(DEVICE),
    }

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            old_metadata = json.load(f)

        if old_metadata != metadata:

            raise RuntimeError(
                "La configuration existante est différente. "
                "Ne mélange pas les expériences."
            )

        print("Métadonnées existantes vérifiées.")

    else:

        with open(
            METADATA_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        print("Métadonnées sauvegardées.")


# =====================================================
# 7. CHARGEMENT ET VERIFICATION DES RESULTATS
# =====================================================

def load_existing_results():

    rows = []
    completed_keys = set()

    if not DETAILED_PATH.exists():
        return rows, completed_keys

    expected_keys = {
        (seed, method, sigma, image_index)
        for seed in SELECTED_SEEDS
        for method in METHODS
        for sigma in SELECTED_SIGMAS
        for image_index in SELECTED_INDICES
    }

    with open(
        DETAILED_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError("En-têtes CSV incorrects.")

        for row in reader:

            try:

                clean_row = {
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "sigma_relative": float(row["sigma_relative"]),
                    "sigma_input": float(row["sigma_input"]),
                    "image_index": int(row["image_index"]),
                    "distance_l2": float(row["distance_l2"]),
                }

            except (ValueError, TypeError, KeyError):
                raise RuntimeError("Ligne CSV invalide.")

            if not math.isfinite(clean_row["distance_l2"]):
                raise RuntimeError("Distance L2 non finie.")

            key = (
                clean_row["seed"],
                clean_row["method"],
                clean_row["sigma_relative"],
                clean_row["image_index"],
            )

            if key not in expected_keys:
                raise RuntimeError("Résultat incompatible avec le test.")

            if key in completed_keys:
                raise RuntimeError("Résultat dupliqué.")

            expected_sigma_input = (
                clean_row["sigma_relative"] * INPUT_RANGE
            )

            if not np.isclose(
                clean_row["sigma_input"],
                expected_sigma_input,
            ):
                raise RuntimeError("sigma_input incorrect.")

            completed_keys.add(key)
            rows.append(clean_row)

    print("Résultats déjà sauvegardés :", len(rows))

    return rows, completed_keys


# =====================================================
# 8. SAUVEGARDE IMMEDIATE
# =====================================================

def save_one_result(row):

    file_exists = (
        DETAILED_PATH.exists()
        and DETAILED_PATH.stat().st_size > 0
    )

    with open(
        DETAILED_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

        f.flush()
        os.fsync(f.fileno())


# =====================================================
# 9. STATISTIQUES DU TEST
# =====================================================

def generate_statistics(existing_rows):

    grouped = {}

    for row in existing_rows:

        key = (
            row["seed"],
            row["method"],
            row["sigma_relative"],
        )

        grouped.setdefault(key, {})[
            row["image_index"]
        ] = row["distance_l2"]

    seed_rows = []

    for seed in SELECTED_SEEDS:

        for method in METHODS:

            for sigma in SELECTED_SIGMAS:

                key = (seed, method, sigma)

                values_by_image = grouped.get(key, {})

                if set(values_by_image) != set(SELECTED_INDICES):
                    continue

                values = [
                    values_by_image[idx]
                    for idx in SELECTED_INDICES
                ]

                seed_rows.append({
                    "seed": seed,
                    "method": method,
                    "sigma_relative": sigma,
                    "mean_distance_l2": float(np.mean(values)),
                    "number_of_images": len(values),
                })

    seed_fields = [
        "seed",
        "method",
        "sigma_relative",
        "mean_distance_l2",
        "number_of_images",
    ]

    with open(
        SEED_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=seed_fields,
        )

        writer.writeheader()
        writer.writerows(seed_rows)

    # Un seul seed : on ne peut pas calculer
    # un écart-type inter-seeds.

    summary_fields = [
        "method",
        "sigma_relative",
        "mean_across_seeds",
        "std_across_seeds",
        "number_of_seeds",
    ]

    summary_rows = []

    for method in METHODS:

        for sigma in SELECTED_SIGMAS:

            matching = [
                row["mean_distance_l2"]
                for row in seed_rows
                if (
                    row["method"] == method
                    and row["sigma_relative"] == sigma
                )
            ]

            if len(matching) == 1:

                summary_rows.append({
                    "method": method,
                    "sigma_relative": sigma,
                    "mean_across_seeds": matching[0],
                    "std_across_seeds": "",
                    "number_of_seeds": 1,
                })

    with open(
        SUMMARY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("Statistiques du test mises à jour.")


# =====================================================
# 10. EXPERIENCE
# =====================================================

def main():

    print("\n======================================")
    print("TEST MINIMAL DE STABILITE")
    print("======================================")

    print("Device :", DEVICE)
    print("Images :", SELECTED_INDICES)
    print("Seeds :", SELECTED_SEEDS)
    print("Sigmas :", SELECTED_SIGMAS)
    print("Perturbations :", NOISE_SAMPLES_SAFE)
    print("Batch size :", BATCH_SIZE)
    print("Méthodes :", METHODS)
    print("Dossier :", OUTPUT_DIR)

    save_metadata()

    existing_rows, completed_keys = load_existing_results()

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

    if max(SELECTED_INDICES) >= len(test_dataset):
        raise ValueError("Indice hors limites du dataset.")

    for seed in SELECTED_SEEDS:

        print(f"\n========== SEED {seed} ==========")

        model_a = None
        model_b = None

        try:

            model_a = load_model(seed, "A")
            model_b = load_model(seed, "B")

            for sigma_index, sigma_relative in enumerate(SELECTED_SIGMAS):

                sigma_input = sigma_relative * INPUT_RANGE

                for method_index, method in enumerate(METHODS):

                    print(
                        f"\nSigma={sigma_relative} | "
                        f"Méthode={method}"
                    )

                    for image_position, image_index in enumerate(
                        SELECTED_INDICES
                    ):

                        key = (
                            seed,
                            method,
                            sigma_relative,
                            image_index,
                        )

                        if key in completed_keys:
                            print(
                                f"Image {image_index} déjà calculée."
                            )
                            continue

                        # Graines reproductibles et indépendantes
                        # pour les modèles A et B.

                        image_seed = (
                            20260000
                            + seed * 1_000_000
                            + sigma_index * 10_000
                            + method_index * 1_000
                            + image_position
                        )

                        generator_a = torch.Generator(device=DEVICE)
                        generator_b = torch.Generator(device=DEVICE)

                        generator_a.manual_seed(image_seed)
                        generator_b.manual_seed(image_seed + 1)

                        image, label = test_dataset[image_index]

                        images = image.unsqueeze(0).to(DEVICE)

                        labels = torch.tensor(
                            [label],
                            dtype=torch.long,
                            device=DEVICE,
                        )

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

                        distance = torch.norm(
                            (map_a - map_b).reshape(1, -1),
                            p=2,
                            dim=1,
                        ).item()

                        if not math.isfinite(distance):
                            raise RuntimeError("Distance L2 non finie.")

                        row = {
                            "seed": seed,
                            "method": method,
                            "sigma_relative": sigma_relative,
                            "sigma_input": sigma_input,
                            "image_index": image_index,
                            "distance_l2": distance,
                        }

                        save_one_result(row)

                        existing_rows.append(row)
                        completed_keys.add(key)

                        print(
                            f"Image {image_index} sauvegardée | "
                            f"Distance L2 = {distance:.6f}"
                        )

                        del (
                            images,
                            labels,
                            map_a,
                            map_b,
                            generator_a,
                            generator_b,
                        )

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        finally:

            if model_a is not None:
                del model_a

            if model_b is not None:
                del model_b

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    generate_statistics(existing_rows)

    total_expected = (
        len(SELECTED_SEEDS)
        * len(METHODS)
        * len(SELECTED_SIGMAS)
        * len(SELECTED_INDICES)
    )

    total_completed = len(completed_keys)

    print("\n======================================")
    print(
        "Résultats sauvegardés :",
        total_completed,
        "/",
        total_expected,
    )

    if total_completed == total_expected:
        print("TEST MINIMAL TERMINE.")
    else:
        print("TEST INCOMPLET : relancer pour reprendre.")

    print("Résultats :", OUTPUT_DIR)
    print("======================================")


if __name__ == "__main__":
    main()
