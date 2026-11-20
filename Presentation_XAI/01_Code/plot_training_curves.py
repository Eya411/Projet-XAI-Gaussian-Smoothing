import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# Dossiers
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Récupération des six historiques
files = sorted(LOG_DIR.glob("resnet34_cifar_N8000_*.csv"))
if len(files) != 6:
    raise ValueError(
        f"6 fichiers attendus, mais {len(files)} trouvés."
    )
histories = {}
for file in files:
    df = pd.read_csv(file)
    required = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "learning_rate"
    ]
    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Colonne {column} absente dans {file.name}"
            )
    if len(df) != 30:
        raise ValueError(
            f"{file.name} contient {len(df)} époques au lieu de 30."
        )
    histories[file.stem] = df
    print(
        f"{file.name} : {len(df)} époques vérifiées"
    )
# 1. Courbes individuelles : Loss et Accuracy
for name, df in histories.items():
    # Loss
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_loss.png",
        dpi=300
    )
    plt.close()
    # Accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["epoch"],
        df["train_accuracy"],
        label="Train Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training Accuracy - {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_accuracy.png",
        dpi=300
    )
    plt.close()
# 2. Comparaison des six modèles : Loss
plt.figure(figsize=(12, 6))
for name, df in histories.items():
    plt.plot(
        df["epoch"],
        df["train_loss"],
        label=name
    )
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Comparison of Training Loss - Six ResNet34 Models")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "comparison_training_loss.png",
    dpi=300
)
plt.close()
# 3. Comparaison des six modèles : Accuracy
plt.figure(figsize=(12, 6))
for name, df in histories.items():
    plt.plot(
        df["epoch"],
        df["train_accuracy"],
        label=name
    )
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("Comparison of Training Accuracy - Six ResNet34 Models")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "comparison_training_accuracy.png",
    dpi=300
)
plt.close()
print("\n" + "=" * 55)
print("GENERATION TERMINEE")
print("=" * 55)
print("Nombre de modèles :", len(histories))
print("Nombre de courbes individuelles :", len(histories) * 2)
print("Nombre de courbes comparatives : 2")
print("Dossier de sauvegarde :", OUTPUT_DIR)
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file.name)
