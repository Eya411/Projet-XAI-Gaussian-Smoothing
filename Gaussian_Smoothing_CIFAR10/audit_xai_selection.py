
import csv

from torchvision.datasets import CIFAR10

from config import DATA_DIR
from xai_config import (
    TEST_IMAGE_INDICES,
    XAI_RESULTS_DIR,
    TEST_DATASET_SIZE,
)

# Charger le jeu de test CIFAR-10 sans le télécharger
dataset = CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False
)

print("Nombre total d'images :", len(dataset))

assert len(dataset) == TEST_DATASET_SIZE, (
    "La taille du jeu de test ne correspond pas à 10000 images."
)

assert len(TEST_IMAGE_INDICES) == 32
assert len(set(TEST_IMAGE_INDICES)) == 32

rows = []

print("\nImages sélectionnées :")

for index in TEST_IMAGE_INDICES:
    image, label = dataset[index]
    class_name = dataset.classes[label]

    rows.append({
        "image_index": index,
        "label_id": label,
        "class_name": class_name
    })

    print(
        f"Index : {index:4d} | "
        f"Classe : {class_name:10s} | "
        f"Label : {label}"
    )

output_file = XAI_RESULTS_DIR / "selected_test_images.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["image_index", "label_id", "class_name"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nAudit terminé.")
print("Nombre d'images vérifiées :", len(rows))
print("Fichier enregistré :", output_file)
