"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, num_channels=1):
        """Initialize network.

        Args:
            num_channels: 1 for MNIST (grayscale), 3 for CIFAR-10 (RGB)
        """
        super(Net, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # MNIST: 28x28 -> after convs: 16*4*4 = 256
        # CIFAR-10: 32x32 -> after convs: 16*5*5 = 400
        fc1_input = 16 * 4 * 4 if num_channels == 1 else 16 * 5 * 5
        self.fc1 = nn.Linear(fc1_input, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten dynamically based on num_channels
        if self.num_channels == 1:
            x = x.view(-1, 16 * 4 * 4)
        else:
            x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

# MNIST transforms: grayscale normalization
mnist_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

# CIFAR-10 transforms: RGB normalization
cifar_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch, dataset="mnist"):
    """Apply transforms to the partition from FederatedDataset."""
    transforms = mnist_transforms if dataset.lower() == "mnist" else cifar_transforms
    # MNIST uses "image" key, CIFAR-10 uses "img" key
    image_key = "image" if dataset.lower() == "mnist" else "img"

    # Apply transforms and normalize key to "image" for consistency
    transformed_images = [transforms(img) for img in batch[image_key]]

    # Create new batch with standardized "image" key
    result = batch.copy()
    result["image"] = transformed_images

    # Remove the original key if it was "img" (CIFAR-10)
    if image_key == "img" and "img" in result:
        del result["img"]

    return result


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioning: str = "iid",
    dirichlet_alpha: float = 0.5,
    dataset: str = "mnist"
):
    """Load partition data with configurable partitioning strategy.

    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        batch_size: Batch size for dataloaders
        partitioning: "iid" or "noniid"
        dirichlet_alpha: Alpha parameter for Dirichlet partitioning
        dataset: "mnist" or "cifar10"
    """
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Create partitioner based on configuration
        if partitioning.lower() == "noniid":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=dirichlet_alpha,
            )
        else:  # iid
            partitioner = IidPartitioner(num_partitions=num_partitions)

        # Load appropriate dataset
        dataset_name = "mnist" if dataset.lower() == "mnist" else "uoft-cs/cifar10"
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders with dataset-specific transforms
    partition_train_test = partition_train_test.with_transform(
        lambda batch: apply_transforms(batch, dataset)
    )
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(dataset="mnist"):
    """Load test set and return dataloader.

    Args:
        dataset: "mnist" or "cifar10"
    """
    # Load entire test set
    if dataset.lower() == "mnist":
        test_dataset = load_dataset("mnist", split="test")
    else:
        test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    dataset_obj = test_dataset.with_format("torch").with_transform(
        lambda batch: apply_transforms(batch, dataset)
    )
    return DataLoader(dataset_obj, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
