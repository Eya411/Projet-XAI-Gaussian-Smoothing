from pathlib import Path
import torch
# ==========================================
# CHEMINS DU PROJET
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
# Création automatique des dossiers
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================
# DATASET
# ==========================================
DATASET_NAME = "CIFAR10"
NUM_CLASSES = 10
IMAGE_SIZE = 32
# Normalisation utilisée dans l'expérience CIFAR-10
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
# ==========================================
# ARCHITECTURE
# ==========================================
MODEL_NAME = "resnet34"
PRETRAINED = False
# ==========================================
# PARAMETRES D'ENTRAINEMENT
# ==========================================
EPOCHS = 30
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
SCHEDULER = "CosineAnnealingLR"
# ==========================================
# EXPERIENCE : DEUX MODELES SUR DEUX SPLITS
# ==========================================
TRAIN_SIZES = [8000, 14000, 25000]
SEEDS = [10007, 5678, 12345]
MODEL_PARTS = ["A", "B"]
# ==========================================
# PARAMETRES TECHNIQUES
# ==========================================
# Valeur provisoire pour le premier test GPU.
# Elle ne doit pas être présentée comme une valeur
# officielle de l'article.
BATCH_SIZE = 64
# Sous Windows, commencer avec 0 pour éviter
# les problèmes de multiprocessing.
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ==========================================
# AFFICHAGE DE LA CONFIGURATION
# ==========================================
def print_config():
    print("=" * 55)
    print("CONFIGURATION DU PROJET XAI - CIFAR-10")
    print("=" * 55)
    print("Dataset :", DATASET_NAME)
    print("Nombre de classes :", NUM_CLASSES)
    print("Taille des images :", IMAGE_SIZE)
    print("Architecture :", MODEL_NAME)
    print("Pretrained :", PRETRAINED)
    print("Epochs :", EPOCHS)
    print("Learning rate :", LEARNING_RATE)
    print("Momentum :", MOMENTUM)
    print("Weight decay :", WEIGHT_DECAY)
    print("Scheduler :", SCHEDULER)
    print("Tailles des splits :", TRAIN_SIZES)
    print("Seeds :", SEEDS)
    print("Batch size provisoire :", BATCH_SIZE)
    print("Workers :", NUM_WORKERS)
    print("Device :", DEVICE)
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
    print("=" * 55)
if __name__ == "__main__":
    print_config()
