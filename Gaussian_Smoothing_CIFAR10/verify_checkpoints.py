import torch
from pathlib import Path
root = Path("Gaussian_Smoothing_CIFAR10/checkpoints")
folders = sorted(
    root.glob("resnet34_cifar_N8000_*")
)
print("=" * 65)
print("VERIFICATION STRUCTURELLE DES CHECKPOINTS")
print("=" * 65)
for folder in folders:
    path = folder / "final_model.pth"
    print("\n", folder.name)
    if not path.exists():
        print("ERREUR : final_model.pth absent")
        continue
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        print("Chargement CPU : OK")
        print("Nombre de tenseurs :", len(state))
        if "conv1.weight" in state:
            print("Conv1 :", tuple(state["conv1.weight"].shape))
        if "fc.weight" in state:
            print("FC :", tuple(state["fc.weight"].shape))
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            blocks = set()
            for key in state:
                if key.startswith(layer + "."):
                    parts = key.split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        blocks.add(int(parts[1]))
            print(f"{layer} : {len(blocks)} blocs")
        if "fc.weight" in state:
            print(
                "Nombre de classes détecté :",
                state["fc.weight"].shape[0]
            )
    except Exception as e:
        print("ERREUR :", str(e))
print("\n" + "=" * 65)
