from tqdm import tqdm
import torchvision
import os


os.makedirs('cifar_images', exist_ok=True)

for i in range(10):
    os.makedirs(f"cifar_images/train/{i}", exist_ok=True)
    os.makedirs(f"cifar_images/val/{i}", exist_ok=True)

train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
for i, (x, y) in enumerate(tqdm(train_data)):
    x.save(f"cifar_images/train/{y}/{i:05}.png")

val_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
for i, (x, y) in enumerate(tqdm(val_data)):
    x.save(f"cifar_images/val/{y}/{i:05}.png")
