from torchvision import datasets
root = "./dataset"
train_dataset = datasets.CIFAR10(
    root=root,
    train=True,
    download=False
)
test_dataset = datasets.CIFAR10(
    root=root,
    train=False,
    download=False
)
print("DATASET CIFAR-10 VERIFIE")
print("Nombre images entrainement :", len(train_dataset))
print("Nombre images test :", len(test_dataset))
print("Nombre de classes :", len(train_dataset.classes))
print("Classes :", train_dataset.classes)
print("Taille première image :", train_dataset[0][0].size)
print("Label première image :", train_dataset[0][1])
