
import random
import csv
from torchvision import datasets

from config import (
    DATA_DIR,
    TRAIN_SIZES,
    SEEDS,
    RESULTS_DIR
)


def create_split_indices(dataset_size, train_size, part, seed):
    """
    Génère un découpage reproductible à partir de la seed.

    part = 0 : modèle A
    part = 1 : modèle B
    """

    if train_size * 2 > dataset_size:
        raise ValueError(
            "La taille des deux splits dépasse le dataset."
        )

    indices = list(range(dataset_size))

    rng = random.Random(seed)
    rng.shuffle(indices)

    start = train_size * part
    end = train_size * (part + 1)

    return indices[start:end]


def verify_split(indices_A, indices_B, train_size, dataset_size):

    assert len(indices_A) == train_size
    assert len(indices_B) == train_size

    assert len(set(indices_A)) == train_size
    assert len(set(indices_B)) == train_size

    overlap = set(indices_A).intersection(set(indices_B))

    assert len(overlap) == 0, (
        "ERREUR : les deux ensembles contiennent des images communes."
    )

    assert min(indices_A + indices_B) >= 0
    assert max(indices_A + indices_B) < dataset_size

    return len(overlap)


def main():

    print("=" * 60)
    print("VERIFICATION DES SPLITS CIFAR-10")
    print("=" * 60)

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False
    )

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False
    )

    dataset_size = len(train_dataset)
    test_size = len(test_dataset)

    print("Images d'entrainement :", dataset_size)
    print("Images de test :", test_size)

    records = []
    all_splits = {}

    for train_size in TRAIN_SIZES:

        print("\nTaille de chaque split :", train_size)

        for seed in SEEDS:

            # Création des deux sous-ensembles
            indices_A = create_split_indices(
                dataset_size,
                train_size,
                part=0,
                seed=seed
            )

            indices_B = create_split_indices(
                dataset_size,
                train_size,
                part=1,
                seed=seed
            )

            # Sauvegarde temporaire pour comparer les seeds
            all_splits[(train_size, seed)] = (
                indices_A,
                indices_B
            )

            # Vérification des propriétés
            overlap = verify_split(
                indices_A,
                indices_B,
                train_size,
                dataset_size
            )

            used = len(set(indices_A + indices_B))
            unused = dataset_size - used

            assert len(indices_A) == train_size
            assert len(indices_B) == train_size
            assert overlap == 0

            print(
                f"Seed {seed} | "
                f"A={len(indices_A)} | "
                f"B={len(indices_B)} | "
                f"Communes={overlap} | "
                f"Non utilisees={unused}"
            )

            records.append({
                "train_size": train_size,
                "seed": seed,
                "model_A_count": len(indices_A),
                "model_B_count": len(indices_B),
                "overlap": overlap,
                "unused_train_images": unused,
                "test_count": test_size,
                "status": "PASS"
            })

    # Vérifier que les différentes seeds produisent
    # des découpages différents pour chaque taille
    for train_size in TRAIN_SIZES:

        split_1 = all_splits[(train_size, SEEDS[0])]

        for seed in SEEDS[1:]:

            split_2 = all_splits[(train_size, seed)]

            assert split_1 != split_2, (
                f"Découpage identique pour les seeds "
                f"{SEEDS[0]} et {seed}, taille {train_size}"
            )

    print("\nVerification des seeds : PASS")

    # Enregistrement du rapport CSV
    output_file = RESULTS_DIR / "split_audit.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 60)
    print("TOUTES LES VERIFICATIONS SONT REUSSIES")
    print("Nombre de configurations :", len(records))
    print("Jeu de test commun :", test_size, "images")
    print("Rapport sauvegarde :", output_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
