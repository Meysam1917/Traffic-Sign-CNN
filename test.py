from sklearn.metrics import classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from model import TrafficSignCNN
from dataset import TrafficSignDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open("data/test.pickle", "rb") as f:
    test_data = pickle.load(f)

with open("data/train.pickle", "rb") as f:
    train_data = pickle.load(f)

test_dataset = TrafficSignDataset(
    test_data["features"],
    test_data["labels"]
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(set(train_data["labels"]))
model = TrafficSignCNN(num_classes).to(device)

model.load_state_dict(torch.load("traffic_sign_cnn.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")


all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_preds)
print(cm.shape)


plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", cbar=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Traffic Sign Classifier")
plt.show()


class_names = [str(i) for i in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=class_names))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
