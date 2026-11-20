
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    NUM_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TRAIN_SIZES,
    SEEDS,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
)


# ==========================================================
# 1. REPRODUCTIBILITE
# ==========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# 2. CONSTRUCTION DES SPLITS
# ==========================================================

def create_split_indices(train_size, dataset_size, seed):

    if train_size * 2 > dataset_size:
        raise ValueError(
            "La taille des deux splits dépasse le dataset."
        )

    indices = list(range(dataset_size))

    # Découpage reproductible propre à chaque seed
    rng = random.Random(seed)
    rng.shuffle(indices)

    indices_A = indices[:train_size]
    indices_B = indices[train_size:2 * train_size]

    assert len(indices_A) == train_size
    assert len(indices_B) == train_size

    assert len(set(indices_A).intersection(indices_B)) == 0

    return indices_A, indices_B


# ==========================================================
# 3. CONSTRUCTION DU MODELE RESNET34
# ==========================================================


def create_model():

    # ResNet34 adapte aux images CIFAR-10 de 32x32
    # Entrainement depuis zero, sans poids ImageNet
    model = models.resnet34(weights=None)

    # Premiere convolution : 3x3 au lieu de 7x7
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # Suppression du MaxPool initial
    model.maxpool = nn.Identity()

    # Classification en 10 classes
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, NUM_CLASSES)

    return model


# ==========================================================
# 4. CHARGEMENT DES DONNEES
# ==========================================================

def load_datasets():

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD
        )
    ])

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform
    )

    return train_dataset, test_dataset


# ==========================================================
# 5. EVALUATION
# ==========================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ==========================================================
# 6. ENTRAINEMENT D'UN MODELE
# ==========================================================

def train_one_model(
    train_dataset,
    test_dataset,
    train_indices,
    train_size,
    seed,
    part,
    epochs
):

    # Reproductibilité de l'initialisation et de l'entraînement
    set_seed(seed)

    model = create_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0
    )

    # Sous-ensemble d'entraînement
    subset = Subset(
        train_dataset,
        train_indices
    )

    # Générateur reproductible pour le mélange des batches
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 2024)

    train_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator
    )

    # Jeu de test commun
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    experiment_name = (
        f"resnet34_cifar_N{train_size}_seed{seed}_model{part}"
    )
   

    checkpoint_folder = CHECKPOINT_DIR / experiment_name

    checkpoint_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = LOGS_DIR / f"{experiment_name}.csv"

    print("\n" + "=" * 65)
    print("DEBUT DE L'ENTRAINEMENT")
    print("Configuration :", experiment_name)
    print("Nombre d'images :", len(subset))
    print("Nombre d'epochs :", epochs)
    print("Batch size :", BATCH_SIZE)
    print("Learning rate :", LEARNING_RATE)
    print("Device :", DEVICE)
    print("=" * 65)

    # Historique des métriques
    with open(
        log_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "learning_rate"
        ])

        for epoch in range(1, epochs + 1):

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total

            train_accuracy = (
                100.0 * correct / total
            )

            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step()

            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                current_lr
            ])

            f.flush()

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"| Loss: {train_loss:.4f} "
                f"| Accuracy: {train_accuracy:.2f}% "
                f"| LR: {current_lr:.6f}"
            )

            # Sauvegarde du dernier état
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_size": train_size,
                "seed": seed,
                "part": part,
                "train_indices": train_indices,
                "batch_size": BATCH_SIZE,
                "epochs": epochs
            }

            torch.save(
                checkpoint,
                checkpoint_folder / "last_checkpoint.pth"
            )

    # ======================================================
    # 7. EVALUATION FINALE
    # ======================================================

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nRESULTAT FINAL")
    print("Test loss :", round(test_loss, 4))
    print("Test accuracy :", round(test_accuracy, 2), "%")

    # Sauvegarde du modèle final
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_size": train_size,
            "seed": seed,
            "part": part,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_indices": train_indices
        },
        checkpoint_folder / "final_model.pth"
    )

    # ======================================================
    # 8. ENREGISTREMENT DES RESULTATS
    # ======================================================

    final_result_file = RESULTS_DIR / "training_results.csv"

    file_exists = (
        final_result_file.exists()
        and final_result_file.stat().st_size > 0
    )

    with open(
        final_result_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "model",
                "train_size",
                "seed",
                "part",
                "epochs",
                "batch_size",
                "test_loss",
                "test_accuracy"
            ])

        writer.writerow([
            "resnet34_cifar",
            train_size,
            seed,
            part,
            epochs,
            BATCH_SIZE,
            test_loss,
            test_accuracy
        ])

    print("\nFichiers sauvegardes :")
    print("Checkpoint :", checkpoint_folder)
    print("Log :", log_file)
    print("Resultats :", final_result_file)

    print("=" * 65)


# ==========================================================
# 9. PROGRAMME PRINCIPAL
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Entrainement ResNet34 sur CIFAR-10"
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
        choices=TRAIN_SIZES
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=SEEDS
    )

    parser.add_argument(
        "--part",
        type=str,
        required=True,
        choices=["A", "B"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs doit être supérieur ou égal à 1")

    print("\nChargement de CIFAR-10...")

    train_dataset, test_dataset = load_datasets()

    print("Images train :", len(train_dataset))
    print("Images test :", len(test_dataset))

    # Création des splits avec la seed de l'expérience
    indices_A, indices_B = create_split_indices(
        args.size,
        len(train_dataset),
        args.seed
    )

    if args.part == "A":
        selected_indices = indices_A
    else:
        selected_indices = indices_B

    print("Seed :", args.seed)
    print("Split sélectionné :", args.part)
    print("Nombre d'images :", len(selected_indices))

    train_one_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_indices=selected_indices,
        train_size=args.size,
        seed=args.seed,
        part=args.part,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
