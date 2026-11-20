
# ============================================================
# GENERATION DES RESULTATS DE FIDELITE - VERSION SECURISEE
# Projet XAI : Ye & Farnia (2025), adaptation CIFAR-10
#
# Comparaison :
# carte originale (sigma = 0)
# contre
# carte moyenne obtenue avec bruit gaussien
#
# Sauvegarde CSV apres chaque image.
# Reprise automatique apres interruption.
# Aucun stockage des cartes bruitees.
# ============================================================

import os
import csv
import math
import gc
import json
import random
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

from xai_config import (
    SEEDS,
    MODEL_PARTS,
    TEST_IMAGE_INDICES,
    METHODS,
    IG_STEPS,
    STABILITY_SIGMAS,
    INPUT_RANGE,
    EXPLANATIONS_DIR,
)


# ============================================================
# 1. CONFIGURATION DE L'EXPERIENCE
# ============================================================

# Version adaptee : 20 perturbations par sigma.
# L'article de reference utilise 100 perturbations.

NOISE_SAMPLES = 20

# Experience complete sur les 8 memes images que la stabilite.
# Les resultats du pilote (seed 10007, modele A) sont conserves
# et reutilises automatiquement par le mecanisme de reprise.

PILOT_MODE = False

EXPERIMENT_IMAGES = list(TEST_IMAGE_INDICES[:8])

# Repertoire distinct : ne touche pas aux resultats de stabilite.

FIDELITY_DIR = (
    Path("results")
    / "phase6_final"
    / "fidelity_common_sigma"
)

FIDELITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAILED_PATH = (
    FIDELITY_DIR
    / "fidelity_per_image.csv"
)

CONFIG_PATH = (
    FIDELITY_DIR
    / "experiment_config.json"
)

FIELDNAMES = [
    "seed",
    "part",
    "method",
    "sigma_relative",
    "sigma_input",
    "image_index",
    "distance_l2",
]


# ============================================================
# 2. CONFIGURATION ACTIVE
# ============================================================

SELECTED_SEEDS = list(SEEDS)
SELECTED_PARTS = list(MODEL_PARTS)
SELECTED_INDICES = list(EXPERIMENT_IMAGES)
SELECTED_METHODS = list(METHODS)
SELECTED_SIGMAS = list(STABILITY_SIGMAS)


# ============================================================
# 3. VERIFICATIONS INITIALES
# ============================================================

def validate_configuration():

    if NOISE_SAMPLES < 1:
        raise ValueError("NOISE_SAMPLES doit etre positif.")

    if not SELECTED_INDICES:
        raise ValueError("Aucune image selectionnee.")

    if not SELECTED_SIGMAS:
        raise ValueError("Aucun sigma selectionne.")

    if not all(np.isfinite(s) and s >= 0 for s in SELECTED_SIGMAS):
        raise ValueError("Valeur sigma invalide.")

    if not np.isclose(INPUT_RANGE, 2.0):
        raise RuntimeError(
            "INPUT_RANGE different de la configuration attendue."
        )

    if not torch.cuda.is_available():
        print("Attention : CUDA indisponible.")
        print("Calcul sur :", DEVICE)

    print("\nConfiguration de fidelite")
    print("-------------------------")
    print("Device :", DEVICE)
    print("Seeds :", SELECTED_SEEDS)
    print("Modeles :", SELECTED_PARTS)
    print("Methodes :", SELECTED_METHODS)
    print("Sigmas :", SELECTED_SIGMAS)
    print("Images :", len(SELECTED_INDICES))
    print("Perturbations :", NOISE_SAMPLES)
    print("Mode pilote :", PILOT_MODE)


# ============================================================
# 4. ARCHITECTURE ET CHARGEMENT DES MODELES
# ============================================================

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

        for key in [
            "model_state_dict",
            "state_dict",
            "model",
        ]:

            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):

                state = checkpoint[key]
                break

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    print(
        f"Modele charge : seed={seed}, part={part}"
    )

    return model


# ============================================================
# 5. METHODES D'EXPLICATION
# ============================================================

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

    for alpha in np.linspace(0.0, 1.0, IG_STEPS):

        interpolated = (
            baseline + alpha * (images - baseline)
        ).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        outputs = model(interpolated)

        scores = outputs.gather(
            1,
            labels.view(-1, 1),
        ).sum()

        scores.backward()

        total_gradient += interpolated.grad.detach()

    return (
        (images - baseline)
        * total_gradient
        / IG_STEPS
    )


def compute_explanation(model, images, labels, method):

    if method == "SimpleGrad":

        return simple_gradient(
            model,
            images,
            labels,
        )

    elif method == "IntegratedGradients":

        return integrated_gradients(
            model,
            images,
            labels,
        )

    else:

        raise ValueError(
            f"Methode inconnue : {method}"
        )


# ============================================================
# 6. CHARGEMENT DU DATASET
# ============================================================

def load_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZE_MEAN,
            NORMALIZE_STD,
        ),
    ])

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform,
    )

    print(
        "Dataset CIFAR-10 test charge :",
        len(dataset),
    )

    return dataset


# ============================================================
# 7. CHARGEMENT DES CARTES ORIGINALES
# ============================================================

def load_original_maps(seed, part):

    filename = (
        f"resnet34_cifar_N8000_seed{seed}"
        f"_model{part}_original_maps_sigma0.npz"
    )

    path = EXPLANATIONS_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Fichier NPZ introuvable : {path}"
        )

    data = np.load(path)

    expected_indices = np.asarray(
        TEST_IMAGE_INDICES
    )

    saved_indices = data["image_indices"]

    if not np.array_equal(
        saved_indices,
        expected_indices,
    ):

        raise RuntimeError(
            f"Indices d'images incorrects dans {filename}"
        )

    if int(data["seed"]) != int(seed):
        raise RuntimeError("Seed NPZ incorrect.")

    if str(data["part"]) != str(part):
        raise RuntimeError("Part NPZ incorrecte.")

    return data


def get_original_map(data, method, position):

    if method == "SimpleGrad":

        key = "simple_gradient"

    elif method == "IntegratedGradients":

        key = "integrated_gradients"

    else:

        raise ValueError(
            f"Methode inconnue : {method}"
        )

    original = data[key][position]

    return torch.as_tensor(
        original,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)


# ============================================================
# 8. GENERATION DE LA CARTE MOYENNE BRUITEE
# ============================================================

def averaged_noisy_explanation(
    model,
    images,
    labels,
    method,
    sigma,
    generator,
):

    total_map = torch.zeros_like(images)

    for _ in range(NOISE_SAMPLES):

        noise = torch.randn(
            images.shape,
            device=DEVICE,
            dtype=images.dtype,
            generator=generator,
        ) * sigma

        noisy_images = images + noise

        explanation = compute_explanation(
            model,
            noisy_images,
            labels,
            method,
        )

        total_map += explanation

    return total_map / NOISE_SAMPLES


# ============================================================
# 9. VERIFICATION DU CSV ET REPRISE
# ============================================================

def load_existing_results():

    rows = []
    completed_keys = set()

    if not DETAILED_PATH.exists():

        return rows, completed_keys

    expected_keys = {
        (
            seed,
            part,
            method,
            float(sigma),
            image_index,
        )
        for seed in SELECTED_SEEDS
        for part in SELECTED_PARTS
        for method in SELECTED_METHODS
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

            raise RuntimeError(
                "En-tetes CSV incorrects."
            )

        for row in reader:

            if None in row:
                raise RuntimeError(
                    "Ligne CSV incomplete detectee."
                )

            clean_row = {
                "seed": int(row["seed"]),
                "part": row["part"],
                "method": row["method"],
                "sigma_relative": float(
                    row["sigma_relative"]
                ),
                "sigma_input": float(
                    row["sigma_input"]
                ),
                "image_index": int(
                    row["image_index"]
                ),
                "distance_l2": float(
                    row["distance_l2"]
                ),
            }

            if not math.isfinite(
                clean_row["distance_l2"]
            ):

                raise RuntimeError(
                    "Distance L2 non finie."
                )

            key = (
                clean_row["seed"],
                clean_row["part"],
                clean_row["method"],
                clean_row["sigma_relative"],
                clean_row["image_index"],
            )

            if key not in expected_keys:

                raise RuntimeError(
                    "Resultat incompatible avec la configuration actuelle."
                )

            if key in completed_keys:

                raise RuntimeError(
                    "Doublon detecte dans le CSV."
                )

            expected_sigma_input = (
                clean_row["sigma_relative"] * INPUT_RANGE
            )

            if not np.isclose(
                clean_row["sigma_input"],
                expected_sigma_input,
            ):

                raise RuntimeError(
                    "sigma_input incoherent."
                )

            completed_keys.add(key)
            rows.append(clean_row)

    print(
        "Resultats existants charges :",
        len(rows),
    )

    return rows, completed_keys


# ============================================================
# 10. SAUVEGARDE IMMEDIATE
# ============================================================

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


# ============================================================
# 11. CONFIGURATION EXPERIMENTALE
# ============================================================

def save_experiment_config():

    config = {
        "experiment": "Fidelity",
        "dataset": "CIFAR-10",
        "architecture": "ResNet34",
        "seeds": SELECTED_SEEDS,
        "parts": SELECTED_PARTS,
        "methods": SELECTED_METHODS,
        "sigmas_relative": SELECTED_SIGMAS,
        "input_range": INPUT_RANGE,
        "noise_samples": NOISE_SAMPLES,
        "ig_steps": IG_STEPS,
        "number_of_images": len(SELECTED_INDICES),
        "image_indices": SELECTED_INDICES,
        "pilot_mode": PILOT_MODE,
        "metric": "L2(smoothed_map, original_map)",
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )


# ============================================================
# 12. CALCUL PRINCIPAL
# ============================================================

def main():

    validate_configuration()

    save_experiment_config()

    dataset = load_dataset()

    rows, completed_keys = load_existing_results()

    total_expected = (
        len(SELECTED_SEEDS)
        * len(SELECTED_PARTS)
        * len(SELECTED_METHODS)
        * len(SELECTED_SIGMAS)
        * len(SELECTED_INDICES)
    )

    print("\nResultats attendus :", total_expected)
    print("Resultats deja termines :", len(rows))

    for seed in SELECTED_SEEDS:

        for part in SELECTED_PARTS:

            model = load_model(seed, part)

            original_data = load_original_maps(
                seed,
                part,
            )

            for method_index, method in enumerate(
                SELECTED_METHODS
            ):

                for sigma_index, sigma_relative in enumerate(
                    SELECTED_SIGMAS
                ):

                    sigma_input = (
                        float(sigma_relative) * INPUT_RANGE
                    )

                    print(
                        f"\nSeed={seed} | Modele={part} "
                        f"| Methode={method} "
                        f"| Sigma={sigma_relative}"
                    )

                    for image_position, image_index in enumerate(
                        SELECTED_INDICES
                    ):

                        key = (
                            seed,
                            part,
                            method,
                            float(sigma_relative),
                            image_index,
                        )

                        if key in completed_keys:

                            continue

                        image, true_label = dataset[
                            image_index
                        ]

                        images = image.unsqueeze(0).to(
                            DEVICE
                        )

                        labels = torch.tensor(
                            [true_label],
                            dtype=torch.long,
                            device=DEVICE,
                        )

                        reference = get_original_map(
                            original_data,
                            method,
                            image_position,
                        )

                        # Graine deterministe unique par combinaison.
                        part_index = (
                            SELECTED_PARTS.index(part)
                        )

                        image_seed = (
                            20260000
                            + seed * 1_000_000
                            + part_index * 100_000
                            + sigma_index * 10_000
                            + method_index * 1_000
                            + image_position
                        )

                        generator = torch.Generator(
                            device=DEVICE
                        )

                        generator.manual_seed(
                            image_seed
                        )

                        with torch.enable_grad():

                            smoothed_map = averaged_noisy_explanation(
                                model,
                                images,
                                labels,
                                method,
                                sigma_input,
                                generator,
                            )

                            distance = torch.norm(
                                smoothed_map - reference,
                                p=2,
                            ).item()

                        if not math.isfinite(distance):

                            raise RuntimeError(
                                "Distance L2 invalide."
                            )

                        row = {
                            "seed": seed,
                            "part": part,
                            "method": method,
                            "sigma_relative": float(
                                sigma_relative
                            ),
                            "sigma_input": float(
                                sigma_input
                            ),
                            "image_index": image_index,
                            "distance_l2": float(distance),
                        }

                        save_one_result(row)

                        completed_keys.add(key)
                        rows.append(row)

                        print(
                            f"Image {image_index} : "
                            f"L2 = {distance:.8f}"
                        )

                        del (
                            images,
                            labels,
                            reference,
                            smoothed_map,
                        )

                        gc.collect()

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            del model
            del original_data

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n================================")
    print("EXPERIENCE TERMINEE")
    print("================================")
    print("Resultats enregistres :", len(rows))
    print("Resultats attendus :", total_expected)
    print("CSV :", DETAILED_PATH)


# ============================================================
# 13. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()

