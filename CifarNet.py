# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) #Conv2d(input channel크기, output channel크기, 필터(nxn))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2) #minpooling?
        self.pool2 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 5) #Conv2d의 output channel이 아래의 Linear함수의 파라미터에 영향
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))        
        x = self.bn2(self.pool2(F.relu(self.conv2(x))))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.5 ,training=self.training)
        for i in range(2):
            x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def _cifarnet(pretrained=False, path=None):
    model = CifarNet()
    if pretrained:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    return model
