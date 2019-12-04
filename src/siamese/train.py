import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import SiameseDataset
from lossfn import ContrastiveLoss
from network import SiameseNet

device = "cuda:1" if torch.cuda.is_available() else "cpu"

sketch_path = "../model_files/sketch.pkl"
photo_path = "../model_files/photo.pkl"

siamese_dataset = SiameseDataset(sketch_path, photo_path)

train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=256)

net = SiameseNet().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_loss = []
for epoch in range(0, 300):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data[0].to(device), data[1].to(device), \
                            data[2].to(device)
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        running_loss += loss_contrastive.item()
    running_loss = running_loss / len(train_dataloader)
    train_loss.append(running_loss)
    if len(train_loss) > 3 and train_loss[-4] > train_loss[-1]:
        torch.save(net.state_dict(), "../model_files/model.pt")
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, running_loss))
