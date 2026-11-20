
import numpy as np
from pathlib import Path

file_path = (
    Path("results/phase6_final/explanations")
    / "resnet34_cifar_N8000_seed10007_modelA_original_maps_sigma0.npz"
)

print("Fichier :", file_path)
print("Existe :", file_path.exists())

with np.load(file_path) as data:

    print("\nClés disponibles :")
    for key in data.files:
        print(" -", key)

    print("\nDimensions et types :")
    for key in [
        "simple_gradient",
        "integrated_gradients",
        "true_labels",
        "image_indices",
    ]:
        arr = data[key]
        print(
            f"{key}: shape={arr.shape}, "
            f"dtype={arr.dtype}, "
            f"min={arr.min()}, max={arr.max()}"
        )

    print("\nMétadonnées :")
    print("sigma :", data["sigma"].item())
    print("IG steps :", data["ig_steps"].item())
    print("baseline :", data["baseline_value"].item())
    print("seed :", data["seed"].item())
    print("part :", data["part"].item())

    print("\nIndices des images :")
    print(data["image_indices"])

    print("\nLabels réels :")
    print(data["true_labels"])

    for key in ["simple_gradient", "integrated_gradients"]:
        print(
            f"\n{key} contient uniquement des valeurs finies :",
            np.isfinite(data[key]).all()
        )
