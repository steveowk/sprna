import torch.nn as nn

class RnaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(1,64,4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv_layer2 = nn.Conv2d(64,128,4)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv_layer3 = nn.Conv2d(128,256,4)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d((2,2))
        self.flatten_layer = nn.Flatten()
        self.linear_layer1 = nn.Linear(1024,256)
        self.dropout = nn.Dropout(0.5)
        self.linear_layer2 = nn.Linear(256, 64)
        self.linear_layer3 = nn.Linear(64, 32)
        self.linear_layer4 = nn.Linear(32,  1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = x.float()
        x = self.relu(self.bn1(self.conv_layer1(x)))
        x = self.relu(self.bn2(self.conv_layer2(x)))
        x = self.relu(self.bn3(self.conv_layer3(x)))
        x = self.pool(x)
        x = self.flatten_layer(x)
        x = self.relu(self.linear_layer1(x))
        x = self.dropout(self.relu(self.linear_layer2(x)))
        x = self.relu(self.linear_layer3(x))
        x = self.linear_layer4(x)
        x = self.tanh(x)
        return x