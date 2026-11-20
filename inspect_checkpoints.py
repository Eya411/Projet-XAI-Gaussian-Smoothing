import torch
from pathlib import Path
root = Path("Gaussian_Smoothing_CIFAR10/checkpoints")
for folder in sorted(root.glob("resnet34_cifar_N8000_*")):
    checkpoint = folder / "last_checkpoint.pth"
    print("\n" + "=" * 65)
    print(folder.name)
    if not checkpoint.exists():
        print("Checkpoint introuvable")
        continue
    data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )
    if isinstance(data, dict):
        print("Clés disponibles :", list(data.keys()))
        for key, value in data.items():
            if any(word in key.lower() for word in
                   ["loss", "acc", "history", "epoch"]):
                print(key, ":", str(value)[:500])
    else:
        print("Type :", type(data))
