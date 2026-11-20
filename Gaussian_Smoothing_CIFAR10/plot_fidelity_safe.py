
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CHEMINS
# ============================================================

FIDELITY_DIR = (
    Path("results")
    / "phase6_final"
    / "fidelity_safe"
)

INPUT_PATH = FIDELITY_DIR / "fidelity_summary.csv"

PNG_PATH = FIDELITY_DIR / "fidelity_curves.png"
PDF_PATH = FIDELITY_DIR / "fidelity_curves.pdf"


# ============================================================
# 2. CHARGEMENT DES RESULTATS
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "method",
    "sigma_relative",
    "mean_distance_l2",
    "std_across_seeds",
    "n_seeds",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes manquantes : {missing}"
    )

if len(df) != 22:
    raise ValueError(
        f"22 lignes attendues, {len(df)} trouvées."
    )

if not (df["n_seeds"] == 3).all():
    raise ValueError(
        "Les résultats ne contiennent pas tous 3 seeds."
    )


# ============================================================
# 3. PREPARATION DES COURBES
# ============================================================

methods = [
    "SimpleGrad",
    "IntegratedGradients",
]

# Six graduations lisibles sur l'axe horizontal.
# Les 11 points expérimentaux restent tous affichés.
X_TICKS = [
    0.000,
    0.064,
    0.144,
    0.256,
    0.400,
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5.5),
    sharey=False,
)

for ax, method in zip(axes, methods):

    subset = (
        df[df["method"] == method]
        .sort_values("sigma_relative")
    )

    if len(subset) != 11:
        raise ValueError(
            f"11 sigmas attendus pour {method}."
        )

    ax.errorbar(
        subset["sigma_relative"],
        subset["mean_distance_l2"],
        yerr=subset["std_across_seeds"],
        marker="o",
        markersize=5,
        capsize=4,
        linewidth=1.8,
    )

    ax.set_title(
        method,
        fontsize=13,
    )

    ax.set_xlabel(
        "Relative Gaussian noise level (sigma)",
        fontsize=10,
    )

    ax.set_ylabel(
        "Mean L2 distance from original map",
        fontsize=10,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    # Affichage de seulement 6 graduations.
    ax.set_xticks(X_TICKS)

    ax.set_xticklabels(
        [f"{value:.3f}" for value in X_TICKS],
        rotation=0,
        fontsize=9,
    )

    ax.set_xlim(-0.01, 0.42)


# ============================================================
# 4. TITRE GENERAL ET MISE EN PAGE
# ============================================================

fig.suptitle(
    "Effect of Gaussian Smoothing on Attribution Map Deviation",
    fontsize=13,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)


# ============================================================
# 5. SAUVEGARDE
# ============================================================

fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 6. VERIFICATION
# ============================================================

print("\n====================================")
print("GRAPHIQUES GENERES AVEC SUCCES")
print("====================================")

print("PNG :", PNG_PATH)
print("PDF :", PDF_PATH)

print("Methodes :", methods)
print("Nombre de points par methode : 11")
print("Nombre de graduations X : 6")
print("Nombre de seeds : 3")
print("Barres d'erreur : ecart-type entre les seeds")

print("\nAucune donnee experimentale n'a ete modifiee.")
