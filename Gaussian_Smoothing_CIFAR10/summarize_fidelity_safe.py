
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CHEMINS
# ============================================================

FIDELITY_DIR = (
    Path("results")
    / "phase6_final"
    / "fidelity_safe"
)

INPUT_PATH = FIDELITY_DIR / "fidelity_per_image.csv"

PER_SEED_PATH = FIDELITY_DIR / "fidelity_per_seed.csv"

SUMMARY_MODEL_PATH = (
    FIDELITY_DIR / "fidelity_summary_by_model.csv"
)

SUMMARY_PATH = FIDELITY_DIR / "fidelity_summary.csv"


# ============================================================
# 2. CHARGEMENT ET VERIFICATION
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "seed",
    "part",
    "method",
    "sigma_relative",
    "image_index",
    "distance_l2",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes manquantes : {missing}"
    )

if df.duplicated(
    subset=[
        "seed",
        "part",
        "method",
        "sigma_relative",
        "image_index",
    ]
).any():
    raise ValueError("Des doublons ont été détectés.")

if df["distance_l2"].isna().any():
    raise ValueError("Des distances L2 sont manquantes.")

print("Fichier brut chargé :", len(df), "lignes")


# ============================================================
# 3. MOYENNE SUR LES IMAGES
# ============================================================

per_seed = (
    df.groupby(
        ["seed", "part", "method", "sigma_relative"],
        as_index=False
    )
    .agg(
        mean_distance_l2=("distance_l2", "mean"),
        std_images=("distance_l2", "std"),
        number_of_images=("image_index", "nunique"),
    )
)

if not (per_seed["number_of_images"] == 8).all():
    raise ValueError(
        "Certaines combinaisons ne contiennent pas 8 images."
    )

per_seed.to_csv(PER_SEED_PATH, index=False)

print("Moyennes par seed et modèle :", len(per_seed), "lignes")


# ============================================================
# 4. STATISTIQUES SEPAREES POUR A ET B
# ============================================================

summary_model = (
    per_seed.groupby(
        ["part", "method", "sigma_relative"],
        as_index=False
    )
    .agg(
        mean_distance_l2=("mean_distance_l2", "mean"),
        std_across_seeds=("mean_distance_l2", "std"),
        n_seeds=("seed", "nunique"),
    )
)

summary_model.to_csv(
    SUMMARY_MODEL_PATH,
    index=False
)

print(
    "Statistiques séparées A/B :",
    len(summary_model),
    "lignes"
)


# ============================================================
# 5. MOYENNE A/B AU SEIN DE CHAQUE SEED
# ============================================================

seed_pair = (
    per_seed.groupby(
        ["seed", "method", "sigma_relative"],
        as_index=False
    )
    .agg(
        mean_distance_l2=("mean_distance_l2", "mean"),
        number_of_models=("part", "nunique"),
    )
)

if not (seed_pair["number_of_models"] == 2).all():
    raise ValueError(
        "Une seed ne contient pas exactement les modèles A et B."
    )


# ============================================================
# 6. MOYENNE ET ECART-TYPE ENTRE LES 3 SEEDS
# ============================================================

summary = (
    seed_pair.groupby(
        ["method", "sigma_relative"],
        as_index=False
    )
    .agg(
        mean_distance_l2=("mean_distance_l2", "mean"),
        std_across_seeds=("mean_distance_l2", "std"),
        n_seeds=("seed", "nunique"),
    )
)

if not (summary["n_seeds"] == 3).all():
    raise ValueError(
        "Certaines statistiques ne contiennent pas 3 seeds."
    )

summary.to_csv(SUMMARY_PATH, index=False)


# ============================================================
# 7. AFFICHAGE
# ============================================================

print("\n====================================")
print("AGREGATION TERMINEE")
print("====================================")

print("Moyennes par seed :", PER_SEED_PATH)
print("Statistiques A/B :", SUMMARY_MODEL_PATH)
print("Résumé final :", SUMMARY_PATH)

print("\nRésumé final :")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nNombre de lignes du résumé :", len(summary))
print("Nombre de seeds indépendantes :", summary["n_seeds"].unique())
