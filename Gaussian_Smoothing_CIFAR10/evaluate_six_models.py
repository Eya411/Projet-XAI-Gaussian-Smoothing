import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "phase5_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
# Limiter les threads pour moderer la charge CPU
torch.set_num_threads(2)
print("=" * 65)
print("PHASE 5 - EVALUATION DES SIX MODELES")
print("Appareil utilise :", DEVICE)
print("=" * 65)
# --------------------------------------------------
# DATASET DE TEST ET PRETRAITEMENT
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])
test_dataset = datasets.CIFAR10(
    root=str(PROJECT_ROOT / "dataset"),
    train=False,
    download=False,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
class_names = test_dataset.classes
print("Images de test :", len(test_dataset))
print("Nombre de classes :", len(class_names))
print("Classes :", class_names)
# --------------------------------------------------
# CONSTRUCTION DE RESNET34 POUR CIFAR-10
# --------------------------------------------------
def create_model():
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )
    return model
# --------------------------------------------------
# ANCIENS RESULTATS POUR COMPARAISON
# --------------------------------------------------
reference_results = {}
reference_file = RESULTS_DIR / "training_results.csv"
if reference_file.exists():
    with open(reference_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "resnet34_cifar":
                key = (
                    row["seed"],
                    row["part"]
                )
                reference_results[key] = row
# --------------------------------------------------
# EVALUATION DES SIX CHECKPOINTS
# --------------------------------------------------
checkpoint_folders = sorted(
    CHECKPOINT_DIR.glob("resnet34_cifar_N8000_*")
)
if len(checkpoint_folders) != 6:
    raise ValueError(
        f"6 dossiers attendus, {len(checkpoint_folders)} trouves."
    )
summary_rows = []
for model_number, folder in enumerate(checkpoint_folders, start=1):
    print("\n" + "=" * 65)
    print(f"MODELE {model_number}/6 : {folder.name}")
    print("=" * 65)
    checkpoint_path = folder / "final_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"model_state_dict absent dans {checkpoint_path}"
        )
    model = create_model()
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )
    model.to(DEVICE)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    all_logits = []
    all_probabilities = []
    all_labels = []
    all_predictions = []
    prediction_file = OUTPUT_DIR / f"{folder.name}_predictions.csv"
    with open(
        prediction_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "correct",
            "confidence"
        ])
        with torch.inference_mode():
            for batch_number, (images, labels) in enumerate(
                test_loader, start=1
            ):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    reduction="sum"
                )
                probabilities = torch.softmax(logits, dim=1)
                confidence, predictions = probabilities.max(dim=1)
                total_loss += loss.item()
                total_correct += (
                    predictions == labels
                ).sum().item()
                total_images += labels.size(0)
                all_logits.append(logits.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                for i in range(len(labels)):
                    true_id = int(labels[i].item())
                    pred_id = int(predictions[i].item())
                    writer.writerow([
                        total_images - len(labels) + i,
                        true_id,
                        class_names[true_id],
                        pred_id,
                        class_names[pred_id],
                        int(true_id == pred_id),
                        float(confidence[i].item())
                    ])
                if batch_number % 50 == 0:
                    print(
                        f"Lots {batch_number}/{len(test_loader)} "
                        f"| Images traitees : {total_images}"
                    )
    test_loss = total_loss / total_images
    test_accuracy = 100.0 * total_correct / total_images
    # Sauvegarde des scores pour les prochaines phases XAI
    logits_array = np.concatenate(all_logits, axis=0)
    probabilities_array = np.concatenate(all_probabilities, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    predictions_array = np.concatenate(all_predictions, axis=0)
    np.savez_compressed(
        OUTPUT_DIR / f"{folder.name}_scores.npz",
        logits=logits_array,
        probabilities=probabilities_array,
        true_labels=labels_array,
        predictions=predictions_array
    )
    seed = folder.name.split("_seed")[1].split("_")[0]
    part = folder.name.split("_model")[-1]
    reference = reference_results.get((seed, part))
    reference_loss = ""
    reference_accuracy = ""
    loss_difference = ""
    accuracy_difference = ""
    if reference:
        reference_loss = float(reference["test_loss"])
        reference_accuracy = float(reference["test_accuracy"])
        loss_difference = test_loss - reference_loss
        accuracy_difference = test_accuracy - reference_accuracy
    summary_rows.append({
        "model": folder.name,
        "test_images": total_images,
        "test_loss_recomputed": test_loss,
        "test_accuracy_recomputed": test_accuracy,
        "test_loss_original": reference_loss,
        "test_accuracy_original": reference_accuracy,
        "loss_difference": loss_difference,
        "accuracy_difference": accuracy_difference
    })
    print("\nRESULTATS :")
    print(f"Test loss     : {test_loss:.6f}")
    print(f"Test accuracy : {test_accuracy:.2f}%")
    if reference:
        print(f"Accuracy initiale : {reference_accuracy:.2f}%")
        print(f"Ecart accuracy    : {accuracy_difference:+.4f}")
    print("Predictions :", prediction_file.name)
    print("Scores sauvegardes :", f"{folder.name}_scores.npz")
    del model
    del checkpoint
# --------------------------------------------------
# ENREGISTREMENT DU BILAN GLOBAL
# --------------------------------------------------
summary_file = OUTPUT_DIR / "phase5_evaluation_summary.csv"
with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary_rows)
print("\n" + "=" * 65)
print("EVALUATION TERMINEE")
print("=" * 65)
print("Modeles evalues :", len(summary_rows))
print("Images de test par modele :", len(test_dataset))
print("Bilan :", summary_file)
for row in summary_rows:
    print(
        row["model"],
        "| Accuracy :",
        f'{row["test_accuracy_recomputed"]:.2f}%'
    )
