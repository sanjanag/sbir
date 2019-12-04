import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, num_classes=125):
        super(SiameseNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 15),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 8),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 3),

            nn.Conv2d(64, 256, 5),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)

            # nn.ReflectionPad2d(1),
            # nn.Conv2d(1, 4, kernel_size=3),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(4),
            #
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(4, 8, kernel_size=3),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(8),
            #
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(8, 8, kernel_size=3),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(8),

        )


        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 125))
            # nn.Linear(8 * 100 * 100, 4096),
            # nn.ReLU(inplace=True),
            #
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            #
            # nn.Linear(4096, 125))

    def forward_once(self, x):
        x = self.cnn1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
