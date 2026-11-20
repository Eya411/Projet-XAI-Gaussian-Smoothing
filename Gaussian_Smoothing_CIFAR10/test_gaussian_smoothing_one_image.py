
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "results" / "phase6_test"
    / "xai_one_image_maps.npz"
)

OUTPUT_DIR = BASE_DIR / "results" / "phase6_gaussian_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Valeurs de sigma pour ce premier test exploratoire
SIGMAS = [0.5, 1.0, 2.0]

METHODS = [
    ("Simple Gradient", "simple_gradient"),
    ("Integrated Gradients", "integrated_gradients"),
    ("SmoothGrad", "smoothgrad")
]

# ==========================================
# 2. CHARGEMENT DES CARTES XAI
# ==========================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

data = np.load(INPUT_FILE)

original_image = data["original_image"]

print("Cartes XAI chargees avec succes.")
print("Dimensions de l'image :", original_image.shape)

# ==========================================
# 3. APPLICATION DU FILTRE GAUSSIEN
# ==========================================

all_results = {}

for method_name, key in METHODS:

    original_map = data[key]

    print("\nMethode :", method_name)

    print("Dimensions de la carte :", original_map.shape)

    all_results[key + "_original"] = original_map

    for sigma in SIGMAS:

        smoothed_map = gaussian_filter(
            original_map,
            sigma=sigma,
            mode="reflect"
        )

        all_results[
            key + "_sigma_" + str(sigma)
        ] = smoothed_map

        print(
            f"Sigma = {sigma} | "
            f"Minimum = {smoothed_map.min():.4f} | "
            f"Maximum = {smoothed_map.max():.4f}"
        )

# ==========================================
# 4. SAUVEGARDE DES CARTES
# ==========================================

np.savez_compressed(
    OUTPUT_DIR / "gaussian_smoothed_maps.npz",
    **all_results
)

print("\nCartes lissees sauvegardees.")

# ==========================================
# 5. VISUALISATION COMPARATIVE
# ==========================================

fig, axes = plt.subplots(
    3, 4,
    figsize=(16, 11)
)

for row, (method_name, key) in enumerate(METHODS):

    # Carte originale
    original_map = data[key]

    axes[row, 0].imshow(
        original_map,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[row, 0].set_title("Originale")

    # Cartes lissees
    for col, sigma in enumerate(SIGMAS, start=1):

        smoothed_map = all_results[
            key + "_sigma_" + str(sigma)
        ]

        # Normalisation pour comparer la forme spatiale
        display_map = smoothed_map.copy()

        maximum = display_map.max()

        if maximum > 0:
            display_map = display_map / maximum

        axes[row, col].imshow(
            display_map,
            cmap="jet",
            vmin=0,
            vmax=1
        )

        axes[row, col].set_title(
            f"Sigma = {sigma}"
        )

    for col in range(4):
        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(
        method_name,
        fontsize=11
    )

fig.suptitle(
    "Effet du lissage gaussien sur les cartes XAI",
    fontsize=15
)

plt.tight_layout()

figure_path = (
    OUTPUT_DIR / "gaussian_smoothing_comparison.png"
)

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 6. FIN
# ==========================================

print("\n========================================")
print("TEST DU LISSAGE GAUSSIEN TERMINE")
print("Resultats :", OUTPUT_DIR)
print("Image :", figure_path)
print("========================================")
