
import numpy as np
from pathlib import Path

from xai_config import (
    SEEDS,
    MODEL_PARTS,
    TEST_IMAGE_INDICES,
    EXPLANATIONS_DIR,
)

print("=" * 60)
print("AUDIT DES EXPLICATIONS ORIGINALES")
print("=" * 60)

reference_indices = np.asarray(TEST_IMAGE_INDICES)
reference_labels = None

total_files = 0
total_maps = 0

for seed in SEEDS:
    for part in MODEL_PARTS:

        name = f"resnet34_cifar_N8000_seed{seed}_model{part}"
        path = EXPLANATIONS_DIR / f"{name}_original_maps_sigma0.npz"

        print(f"\nModèle : {name}")

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as data:

            simple = data["simple_gradient"]
            integrated = data["integrated_gradients"]
            labels = data["true_labels"]
            indices = data["image_indices"]

            assert simple.shape == (32, 3, 32, 32)
            assert integrated.shape == (32, 3, 32, 32)

            assert np.array_equal(indices, reference_indices)

            assert np.isfinite(simple).all()
            assert np.isfinite(integrated).all()

            assert not np.all(simple == 0)
            assert not np.all(integrated == 0)

            if reference_labels is None:
                reference_labels = labels.copy()
            else:
                assert np.array_equal(labels, reference_labels)

            assert data["sigma"].item() == 0.0
            assert data["ig_steps"].item() == 20
            assert data["baseline_value"].item() == 0.0

            print("  Dimensions :", simple.shape)
            print("  Indices identiques : OK")
            print("  Labels identiques : OK")
            print("  Valeurs finies : OK")
            print("  Cartes non nulles : OK")
            print("  Sigma :", data["sigma"].item())

            total_files += 1
            total_maps += len(simple) * 2

print("\n" + "=" * 60)
print("AUDIT TERMINE")
print("Fichiers vérifiés :", total_files)
print("Cartes vérifiées :", total_maps)
print("Tous les contrôles sont passés.")
print("=" * 60)
