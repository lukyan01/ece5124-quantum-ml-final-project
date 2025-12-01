from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64, root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST norm
    ])

    train_ds = datasets.MNIST(root, train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
