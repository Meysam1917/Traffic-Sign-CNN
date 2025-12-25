from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from model import TrafficSignCNN
from dataset import TrafficSignDataset


# load data
with open("data/train.pickle", "rb") as f:
    train_data = pickle.load(f)

with open("data/test.pickle", "rb") as f:
    test_data = pickle.load(f)

if __name__ == "__main__":
    train_dataset = TrafficSignDataset(
        train_data["features"],
        train_data["labels"]
    )

    test_dataset = TrafficSignDataset(
        test_data["features"],
        test_data["labels"]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)




    num_classes = len(set(train_data["labels"]))
    print("Number of classes:", num_classes)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrafficSignCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    print("Output shape:", outputs.shape)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Accuracy: {train_acc:.2f}%")

    torch.save(model.state_dict(), "traffic_sign_cnn.pth")
