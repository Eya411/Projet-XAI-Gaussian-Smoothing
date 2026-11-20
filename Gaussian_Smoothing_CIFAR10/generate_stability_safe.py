
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
    STABILITY_SIGMAS,
    IG_STEPS,
    INPUT_RANGE,
)


# =====================================================
# 1. CONFIGURATION SECURISEE
# =====================================================

N_IMAGES = 8
NOISE_SAMPLES_SAFE = 20
BATCH_SIZE = 1

SELECTED_INDICES = [
    int(x) for x in TEST_IMAGE_INDICES[:N_IMAGES]
]

SELECTED_SEEDS = [int(x) for x in SEEDS]

SELECTED_SIGMAS = [
    float(x) for x in STABILITY_SIGMAS
]

METHODS = [
    "SimpleGrad",
    "IntegratedGradients",
]

OUTPUT_DIR = Path(
    "results/phase6_final/stability_safe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAILED_PATH = (
    OUTPUT_DIR / "stability_per_image.csv"
)

SEED_PATH = (
    OUTPUT_DIR / "stability_per_seed.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "stability_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR / "experiment_config.json"
)

FIELDNAMES = [
    "seed",
    "method",
    "sigma_relative",
    "sigma_input",
    "image_index",
    "distance_l2",
]


# Vérification de la configuration

if len(SELECTED_INDICES) != N_IMAGES:
    raise ValueError(
        "Nombre d'indices incorrect."
    )

if len(set(SELECTED_INDICES)) != N_IMAGES:
    raise ValueError(
        "Les indices des images doivent être uniques."
    )

if len(SELECTED_SEEDS) == 0:
    raise ValueError("Aucun seed défini.")

if len(SELECTED_SIGMAS) == 0:
    raise ValueError("Aucun sigma défini.")

if NOISE_SAMPLES_SAFE <= 0:
    raise ValueError("NOISE_SAMPLES_SAFE doit être positif.")

if BATCH_SIZE != 1:
    raise ValueError("Le batch size doit rester égal à 1.")


# =====================================================
# 2. ARCHITECTURE ET CHARGEMENT DES MODELES
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
        f"Modèle chargé : seed={seed}, part={part}"
    )

    return model


# =====================================================
# 3. METHODES D'EXPLICATION
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

    for alpha in np.linspace(
        0.0,
        1.0,
        IG_STEPS,
    ):

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


# =====================================================
# 4. MOYENNE DES EXPLICATIONS SOUS BRUIT GAUSSIEN
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
            device=DEVICE,
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

    return total_map / NOISE_SAMPLES_SAFE


# =====================================================
# 5. METADONNEES ET CONTROLE DE CONFIGURATION
# =====================================================

def save_metadata():

    metadata = {
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
                "ERREUR : les paramètres actuels "
                "ne correspondent pas à l'expérience "
                "déjà sauvegardée. "
                "Ne pas mélanger les expériences."
            )

        print(
            "Configuration existante vérifiée."
        )

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

        print(
            "Configuration initiale sauvegardée."
        )


# =====================================================
# 6. CHARGEMENT DES RESULTATS EXISTANTS
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

            raise RuntimeError(
                "En-têtes CSV incorrects. "
                "Vérifier le fichier de résultats."
            )

        for row in reader:

            if None in row:
                raise RuntimeError(
                    "Ligne CSV incomplète détectée. "
                    "Ne pas continuer avant vérification."
                )

            try:

                clean_row = {
                    "seed": int(row["seed"]),
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

            except (ValueError, TypeError, KeyError):

                raise RuntimeError(
                    "Une ligne du CSV contient des données invalides."
                )

            if not math.isfinite(
                clean_row["distance_l2"]
            ):

                raise RuntimeError(
                    "Distance L2 non finie détectée."
                )

            key = (
                clean_row["seed"],
                clean_row["method"],
                clean_row["sigma_relative"],
                clean_row["image_index"],
            )

            if key not in expected_keys:

                raise RuntimeError(
                    "Un résultat ne correspond pas "
                    "à la configuration actuelle."
                )

            if key in completed_keys:

                raise RuntimeError(
                    "Résultat dupliqué détecté dans le CSV."
                )

            expected_sigma_input = (
                clean_row["sigma_relative"] * INPUT_RANGE
            )

            if not np.isclose(
                clean_row["sigma_input"],
                expected_sigma_input,
            ):

                raise RuntimeError(
                    "sigma_input incohérent dans le CSV."
                )

            completed_keys.add(key)
            rows.append(clean_row)

    print(
        "Résultats existants chargés :",
        len(rows),
    )

    return rows, completed_keys


# =====================================================
# 7. SAUVEGARDE IMMEDIATE APRES CHAQUE IMAGE
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
# 8. CALCUL DES STATISTIQUES
# =====================================================

def generate_statistics(existing_rows):

    grouped = {}

    for row in existing_rows:

        key = (
            row["seed"],
            row["method"],
            row["sigma_relative"],
        )

        grouped.setdefault(
            key,
            {},
        )[row["image_index"]] = row["distance_l2"]

    seed_rows = []

    for seed in SELECTED_SEEDS:

        for method in METHODS:

            for sigma in SELECTED_SIGMAS:

                key = (
                    seed,
                    method,
                    sigma,
                )

                values_by_image = grouped.get(
                    key,
                    {},
                )

                # Ne garder que les configurations complètes.
                if set(values_by_image.keys()) != set(
                    SELECTED_INDICES
                ):

                    continue

                values = [
                    values_by_image[idx]
                    for idx in SELECTED_INDICES
                ]

                seed_rows.append({

                    "seed": seed,
                    "method": method,
                    "sigma_relative": sigma,
                    "mean_distance_l2": float(
                        np.mean(values)
                    ),
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

    # -------------------------------------------------
    # Moyenne et écart-type entre les seeds
    # -------------------------------------------------

    overall_rows = []

    for method in METHODS:

        for sigma in SELECTED_SIGMAS:

            values = []

            for seed in SELECTED_SEEDS:

                matching = [
                    row["mean_distance_l2"]
                    for row in seed_rows
                    if (
                        row["seed"] == seed
                        and row["method"] == method
                        and row["sigma_relative"] == sigma
                    )
                ]

                if matching:

                    values.append(matching[0])

            # Ne pas calculer de moyenne globale
            # si les trois seeds ne sont pas complets.

            if len(values) != len(SELECTED_SEEDS):

                continue

            overall_rows.append({

                "method": method,
                "sigma_relative": sigma,
                "mean_across_seeds": float(
                    np.mean(values)
                ),
                "std_across_seeds": float(
                    np.std(values, ddof=1)
                ),
                "number_of_seeds": len(values),

            })

    summary_fields = [
        "method",
        "sigma_relative",
        "mean_across_seeds",
        "std_across_seeds",
        "number_of_seeds",
    ]

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
        writer.writerows(overall_rows)

    print(
        "\nConfigurations complètes par seed :",
        len(seed_rows),
    )

    print(
        "Configurations complètes entre seeds :",
        len(overall_rows),
    )


# =====================================================
# 9. EXPERIENCE DE STABILITE AVEC REPRISE AUTOMATIQUE
# =====================================================

def main():

    print("\n======================================")
    print("EXPERIENCE DE STABILITE SECURISEE")
    print("======================================")

    print("Device :", DEVICE)
    print("Images :", len(SELECTED_INDICES))
    print("Seeds :", SELECTED_SEEDS)
    print("Sigmas :", SELECTED_SIGMAS)
    print("Perturbations :", NOISE_SAMPLES_SAFE)
    print("Batch size :", BATCH_SIZE)
    print("Methodes :", METHODS)

    print("Dossier :", OUTPUT_DIR)

    # Vérifier les paramètres avant de commencer.

    save_metadata()

    existing_rows, completed_keys = (
        load_existing_results()
    )

    # -------------------------------------------------
    # Dataset CIFAR-10
    # -------------------------------------------------

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

        raise ValueError(
            "Un indice dépasse la taille du dataset CIFAR-10."
        )

    # -------------------------------------------------
    # Boucle principale
    # -------------------------------------------------

    for seed in SELECTED_SEEDS:

        print(
            f"\n========== SEED {seed} =========="
        )

        model_a = None
        model_b = None

        try:

            model_a = load_model(
                seed,
                "A",
            )

            model_b = load_model(
                seed,
                "B",
            )

            for sigma_index, sigma_relative in enumerate(
                SELECTED_SIGMAS
            ):

                sigma_input = (
                    sigma_relative * INPUT_RANGE
                )

                for method_index, method in enumerate(
                    METHODS
                ):

                    print(
                        f"\nSeed={seed} | "
                        f"Sigma={sigma_relative:.3f} | "
                        f"Méthode={method}"
                    )

                    remaining_indices = [

                        idx

                        for idx in SELECTED_INDICES

                        if (
                            seed,
                            method,
                            sigma_relative,
                            idx,
                        ) not in completed_keys

                    ]

                    if not remaining_indices:

                        print(
                            "Configuration déjà terminée. Ignorée."
                        )

                        continue

                    print(
                        "Images restantes :",
                        len(remaining_indices),
                    )

                    # -------------------------------------------------
                    # Calcul image par image
                    # -------------------------------------------------

                    for image_index in remaining_indices:

                        # Graine déterministe propre à chaque image.
                        # Les générateurs A et B restent indépendants.

                        image_position = (
                            SELECTED_INDICES.index(image_index)
                        )

                        image_seed = (
                            20260000
                            + seed * 1_000_000
                            + sigma_index * 10_000
                            + method_index * 1_000
                            + image_position
                        )

                        generator_a = torch.Generator(
                            device=DEVICE
                        )

                        generator_b = torch.Generator(
                            device=DEVICE
                        )

                        generator_a.manual_seed(
                            image_seed
                        )

                        generator_b.manual_seed(
                            image_seed + 1
                        )

                        image, label = test_dataset[
                            image_index
                        ]

                        images = (
                            image.unsqueeze(0)
                            .to(DEVICE)
                        )

                        labels = torch.tensor(
                            [label],
                            dtype=torch.long,
                            device=DEVICE,
                        )

                        # Explication du modèle A

                        map_a = averaged_noisy_explanation(
                            model_a,
                            images,
                            labels,
                            method,
                            sigma_input,
                            generator_a,
                        )

                        # Explication du modèle B

                        map_b = averaged_noisy_explanation(
                            model_b,
                            images,
                            labels,
                            method,
                            sigma_input,
                            generator_b,
                        )

                        # Distance L2 entre les cartes moyennes

                        distance = torch.norm(
                            (map_a - map_b).reshape(1, -1),
                            p=2,
                            dim=1,
                        ).item()

                        if not math.isfinite(distance):

                            raise RuntimeError(
                                "Distance L2 non finie. "
                                "Calcul interrompu."
                            )

                        row = {

                            "seed": seed,
                            "method": method,
                            "sigma_relative": sigma_relative,
                            "sigma_input": sigma_input,
                            "image_index": image_index,
                            "distance_l2": distance,

                        }

                        # Sauvegarder immédiatement.

                        save_one_result(row)

                        existing_rows.append(row)

                        completed_keys.add((
                            seed,
                            method,
                            sigma_relative,
                            image_index,
                        ))

                        print(
                            f"Image {image_index} sauvegardée | "
                            f"Distance L2={distance:.6f}"
                        )

                        del (
                            images,
                            labels,
                            map_a,
                            map_b,
                            generator_a,
                            generator_b,
                        )

                    print(
                        "Configuration terminée."
                    )

        finally:

            if model_a is not None:

                del model_a

            if model_b is not None:

                del model_b

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # -------------------------------------------------
    # Statistiques finales
    # -------------------------------------------------

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
        "Images/configurations sauvegardées :",
        total_completed,
        "/",
        total_expected,
    )

    if total_completed == total_expected:

        print("EXPERIENCE TERMINEE.")

    else:

        print(
            "EXPERIENCE INCOMPLETE : "
            "relancer le script pour reprendre."
        )

    print(
        "Résultats :",
        OUTPUT_DIR,
    )

    print("======================================")


# =====================================================
# 10. POINT D'ENTREE
# =====================================================

if __name__ == "__main__":

    main()
