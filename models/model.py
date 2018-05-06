# coding:utf-8
'''
python 3.5
pytorch 0.4.0
visdom 0.1.7
torchnet 0.0.2
auther: helloholmes
'''
import torch
import time
from torch import nn
from torchvision.models import vgg16
from torchvision.models import resnet34

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%Y_%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)

class Vgg16(BasicModule):
    def __init__(self, num_class=120):
        super(Vgg16, self).__init__()
        model = vgg16(pretrained=True)
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = False
        num_ftr = model.classifier[0].in_features
        self.classifier = nn.Sequential(nn.Linear(num_ftr, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, num_class))

    def set_requires_grad(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet34(BasicModule):
    def __init__(self, num_class=120):
        super(ResNet34, self).__init__()
        model = resnet34(pretrained=True)
        self.features = nn.Sequential(model.conv1,
                                    model.bn1,
                                    model.relu,
                                    model.maxpool,
                                    model.layer1,
                                    model.layer2,
                                    model.layer3,
                                    model.layer4,
                                    model.avgpool)
        for param in self.features.parameters():
            param.requires_grad = False
        num_ftr = model.fc.in_features
        self.classifier = nn.Sequential(nn.Linear(num_ftr, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, num_class))

    def set_requires_grad(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x