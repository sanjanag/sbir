import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from dataset import SiameseDataset
from network import SiameseNet

device = "cuda:1" if torch.cuda.is_available() else "cpu"

sketch_path = "../model_files/sketch.pkl"
photo_path = "../model_files/photo.pkl"

siamese_dataset = SiameseDataset(sketch_path, photo_path)

test_dataloader = DataLoader(siamese_dataset)
model = SiameseNet().to(device)
model.load_state_dict(torch.load("../model_files/model.pt"))


def imshow(img, text=None, filename=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("../results/" + filename + ".png")


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


for data in test_dataloader:
    _, s, _, _, sl = data
    if sl.numpy()[0] == 103:
        break

result = []
for i, data in enumerate(test_dataloader):
    im, _, _, iml, _ = data
    output1, output2 = model(im.to(device), s.to(device))
    euclidean_distance = F.pairwise_distance(output1, output2).item()
    print('im:{} s:{} Dissimilarity: {:.2f}'.format(iml.numpy()[0],
                                                    sl.numpy()[0],
                                                    euclidean_distance))
    result.append((euclidean_distance, im, iml))

# for i in range(12000):
#     im, _, _, iml, _ = next(dataiter)
#
#     output1, output2 = model(im.to(device), s.to(device))
#     euclidean_distance = F.pairwise_distance(output1, output2).item()
#     print('im:{} s:{} Dissimilarity: {:.2f}'.format(iml.numpy()[0],
#                                                     sl.numpy()[0],
#                                                     euclidean_distance))
#     # imshow(torchvision.utils.make_grid(concatenated),
#     #        'im:{} s:{} Dissimilarity: {:.2f}'.format(iml.numpy()[0],
#     #                                                  sl.numpy()[0],
#     #                                                  euclidean_distance))
#     #
#     result.append((euclidean_distance, im, iml))
sorted_by_distance = sorted(result, key=lambda tup: tup[0])

for i in range(min(100, len(sorted_by_distance))):
    distance, im, iml = sorted_by_distance[i]
    concatenated = torch.cat((im, s), 0) * 0.5 + 0.5
    imshow(torchvision.utils.make_grid(concatenated),
           'im:{} s:{} Dissimilarity: {:.2f}'.format(iml.numpy()[0],
                                                     sl.numpy()[0],
                                                     distance),
           str(i))
