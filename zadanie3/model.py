import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNRegressor(nn.Module):
    def __init__(self, useBatchnorm=True, useDropout=True, dropoutP=0.3):
        super().__init__()
        self.useBatchnorm = useBatchnorm
        self.useDropout = useDropout

        def convBlock(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if useBatchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.conv1 = convBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = convBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = convBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2,2)

        # adaptíve pooling – always 4×4 output -> counter 128x128 bad traning
        self.adaptPool = nn.AdaptiveAvgPool2d((4,4))

        self.flattenDim = 128 * 4 * 4

        self.fc1 = nn.Linear(self.flattenDim, 256)
        self.dropout = nn.Dropout(p=dropoutP) if useDropout else nn.Identity()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))

        x = self.adaptPool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
