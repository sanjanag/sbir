import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classnetwork import Net
import pickle as pkl
from dataset import SketchyDataset

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

train_path = "../data/sketchy/photo/tx_000000000000/"
trainset = SketchyDataset(train_path)
with open("../label_dict", "wb") as f:
    pkl.dump(trainset.label_dict, f)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

model = Net().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
train_loss = []
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    running_loss = running_loss / len(trainloader)
    train_loss.append(running_loss)
    if len(train_loss) > 3 and train_loss[-4] > train_loss[-1]:
        torch.save(model.state_dict(), "../model.pt")
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, running_loss))

torch.save(model.state_dict(), "../model.pt")
