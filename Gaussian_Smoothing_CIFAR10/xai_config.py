
from pathlib import Path
import random

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
)

# ==========================================
# CONFIGURATION DES EXPERIENCES XAI
# ==========================================

# Modèles déjà entraînés
SEEDS = [10007, 12345, 5678]
MODEL_PARTS = ["A", "B"]
TRAIN_SIZE = 8000

# Sélection reproductible des images
N_TEST_IMAGES = 32
TEST_DATASET_SIZE = 10000
SELECTION_SEED = 10007

indices = list(range(TEST_DATASET_SIZE))
random.seed(SELECTION_SEED)
random.shuffle(indices)

TEST_IMAGE_INDICES = indices[:N_TEST_IMAGES]

# Méthodes d'explication
METHODS = ["SimpleGrad", "IntegratedGradients"]
IG_STEPS = 20

# Bruit gaussien
NOISE_SAMPLES = 100

# Stabilité : sigma relatif
STABILITY_SIGMAS = [
    0.00, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.40
]

# Fidélité : grille quadratique
FIDELITY_SIGMAS = [
    0.000, 0.004, 0.016, 0.036,
    0.064, 0.100, 0.144, 0.196,
    0.256, 0.324, 0.400
]

# CIFAR-10 normalisé dans [-1, 1]
INPUT_RANGE = 2.0

# Baseline Integrated Gradients
IG_BASELINE_VALUE = 0.0

# Dossiers de résultats
XAI_RESULTS_DIR = RESULTS_DIR / "phase6_final"
EXPLANATIONS_DIR = XAI_RESULTS_DIR / "explanations"
STABILITY_DIR = XAI_RESULTS_DIR / "stability"
FIDELITY_DIR = XAI_RESULTS_DIR / "fidelity"
PLOTS_DIR = XAI_RESULTS_DIR / "plots"

for directory in [
    XAI_RESULTS_DIR,
    EXPLANATIONS_DIR,
    STABILITY_DIR,
    FIDELITY_DIR,
    PLOTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

print("Configuration XAI chargée.")
print("Seeds modèles :", SEEDS)
print("Graine de sélection :", SELECTION_SEED)
print("Nombre d'images :", len(TEST_IMAGE_INDICES))
print("Indices sélectionnés :", TEST_IMAGE_INDICES)
print("Nombre de bruits :", NOISE_SAMPLES)
print("Sigma stabilité :", len(STABILITY_SIGMAS))
print("Sigma fidélité :", len(FIDELITY_SIGMAS))
