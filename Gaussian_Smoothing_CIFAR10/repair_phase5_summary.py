import csv
from pathlib import Path
results_dir = Path("results")
output_dir = results_dir / "phase5_evaluation"
training_file = results_dir / "training_results.csv"
summary_file = output_dir / "phase5_evaluation_summary.csv"
# 1. Lire les résultats originaux
reference_results = {}
with open(training_file, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("model") == "resnet34_cifar":
            key = (row["seed"], row["part"])
            reference_results[key] = row
# 2. Lire les résultats déjà recalculés
with open(summary_file, "r", newline="", encoding="utf-8-sig") as f:
    summary_rows = list(csv.DictReader(f))
# 3. Compléter les colonnes de comparaison
for row in summary_rows:
    seed = row["model"].split("_seed")[1].split("_")[0]
    part = row["model"].split("_model")[-1]
    reference = reference_results[(seed, part)]
    original_loss = float(reference["test_loss"])
    original_accuracy = float(reference["test_accuracy"])
    recomputed_loss = float(row["test_loss_recomputed"])
    recomputed_accuracy = float(row["test_accuracy_recomputed"])
    row["test_loss_original"] = original_loss
    row["test_accuracy_original"] = original_accuracy
    row["loss_difference"] = recomputed_loss - original_loss
    row["accuracy_difference"] = recomputed_accuracy - original_accuracy
# 4. Réécrire le fichier de synthèse
fieldnames = [
    "model",
    "test_images",
    "test_loss_recomputed",
    "test_accuracy_recomputed",
    "test_loss_original",
    "test_accuracy_original",
    "loss_difference",
    "accuracy_difference"
]
with open(summary_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)
print("Synthèse corrigée avec succès.")
print("Nombre de modèles :", len(summary_rows))
for row in summary_rows:
    print(
        row["model"],
        "| Accuracy initiale :", row["test_accuracy_original"],
        "| Accuracy recalculée :", row["test_accuracy_recomputed"],
        "| Écart :", row["accuracy_difference"]
    )
