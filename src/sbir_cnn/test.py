import pickle as pkl

import torch
from torch.utils.data import DataLoader

from classnetwork import Net
from dataset import SketchyDataset

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

test_path = "../data/sketchy/sketch/tx_000000000000/"

with open("label_dict", "rb") as f:
    label_dict = pkl.load(f)

trainset = SketchyDataset(test_path, label_dict, "png")
testloader = DataLoader(trainset, batch_size=500, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load("../model.pt"))
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print(labels, predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(
    'Accuracy of the network on test images: %d %%' % (100 * correct / total))
