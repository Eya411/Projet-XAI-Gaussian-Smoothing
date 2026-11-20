
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
)

from xai_config import (
    SEEDS,
    MODEL_PARTS,
    IG_STEPS,
    TEST_IMAGE_INDICES,
    EXPLANATIONS_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.cuda.set_device(0)
    print("GPU :", torch.cuda.get_device_name(0))
else:
    torch.set_num_threads(2)
    print("Utilisation CPU")

print("Nombre d'images :", len(TEST_IMAGE_INDICES))
print("Nombre de seeds :", len(SEEDS))
print("Méthodes : SimpleGrad et Integrated Gradients")
print("Sigma : 0")

# ==========================================
# 2. ARCHITECTURE DU MODELE
# ==========================================

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
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def load_model(seed, part):
    name = f"resnet34_cifar_N8000_seed{seed}_model{part}"
    path = CHECKPOINT_DIR / name / "final_model.pth"

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("\nModèle chargé :", name)

    return model, name


# ==========================================
# 3. DATASET CIFAR-10
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD
    )
])

dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

# Vérifier que les indices sont valides
assert all(0 <= i < len(dataset) for i in TEST_IMAGE_INDICES)
assert len(set(TEST_IMAGE_INDICES)) == len(TEST_IMAGE_INDICES)

# ==========================================
# 4. SIMPLE GRAD
# ==========================================

def simple_grad(model, inputs, labels):

    inputs = inputs.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits = model(inputs)

    score = logits.gather(
        1,
        labels.view(-1, 1)
    ).sum()

    score.backward()

    return inputs.grad.detach()


# ==========================================
# 5. INTEGRATED GRADIENTS
# ==========================================

def integrated_grad(model, inputs, labels):

    baseline = torch.zeros_like(inputs)
    total_gradient = torch.zeros_like(inputs)

    for alpha in np.linspace(0, 1, IG_STEPS):

        point = (
            baseline + alpha * (inputs - baseline)
        ).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        logits = model(point)

        score = logits.gather(
            1,
            labels.view(-1, 1)
        ).sum()

        score.backward()

        total_gradient += point.grad.detach()

    attribution = (
        (inputs - baseline) * total_gradient / IG_STEPS
    )

    return attribution.detach()


# ==========================================
# 6. GENERATION DES EXPLICATIONS
# ==========================================

for seed in SEEDS:

    for part in MODEL_PARTS:

        model, model_name = load_model(seed, part)

        simple_maps = []
        integrated_maps = []
        labels_list = []

        for position, image_index in enumerate(TEST_IMAGE_INDICES):

            image, true_label = dataset[image_index]

            inputs = image.unsqueeze(0).to(DEVICE)

            labels = torch.tensor(
                [true_label],
                dtype=torch.long,
                device=DEVICE
            )

            with torch.enable_grad():

                simple = simple_grad(model, inputs, labels)

                integrated = integrated_grad(
                    model,
                    inputs,
                    labels
                )

            simple_maps.append(
                simple[0].cpu().numpy()
            )

            integrated_maps.append(
                integrated[0].cpu().numpy()
            )

            labels_list.append(true_label)

            print(
                f"{model_name} | "
                f"Image {position + 1:02d}/32 | "
                f"Index CIFAR : {image_index} | "
                f"Classe : {dataset.classes[true_label]}"
            )

        # Convertir en tableaux NumPy
        simple_maps = np.stack(simple_maps).astype(np.float32)
        integrated_maps = np.stack(integrated_maps).astype(np.float32)

        labels_array = np.asarray(labels_list, dtype=np.int64)
        indices_array = np.asarray(TEST_IMAGE_INDICES, dtype=np.int64)

        # Sauvegarde des cartes signées originales
        output_file = (
            EXPLANATIONS_DIR /
            f"{model_name}_original_maps_sigma0.npz"
        )

        np.savez_compressed(
            output_file,
            simple_gradient=simple_maps,
            integrated_gradients=integrated_maps,
            true_labels=labels_array,
            image_indices=indices_array,
            sigma=0.0,
            ig_steps=IG_STEPS,
            baseline_value=0.0,
            seed=seed,
            part=part
        )

        print("\nFichier sauvegardé :", output_file)
        print("SimpleGrad :", simple_maps.shape)
        print("Integrated Gradients :", integrated_maps.shape)

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ==========================================
# 7. FIN
# ==========================================

print("\n" + "=" * 55)
print("GENERATION DES EXPLICATIONS ORIGINALES TERMINEE")
print("Nombre de modèles :", len(SEEDS) * len(MODEL_PARTS))
print("Nombre total de cartes :", len(SEEDS) * len(MODEL_PARTS) * len(TEST_IMAGE_INDICES) * 2)
print("Dossier :", EXPLANATIONS_DIR)
