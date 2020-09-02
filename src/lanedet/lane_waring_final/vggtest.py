import torch.nn as nn
import torch
import time
from torchvision.models import *
import torch.backends.cudnn as cudnn
import os


class SE_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.set_device(int('0'))
    print(torch.cuda.current_device())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.rand(size=(8, 3, 224, 224))

    net = SE_VGG(num_classes=10).to(device)
    # net = resnet50(pretrained=False).to(device)
    net.eval()
    x = x.to(device)
    # cudnn.benchmark = True
    # cudnn.fastest = True

    with torch.no_grad():
        t = time.time()
        out = net(x)
        print(time.time() - t)
    # print(out.size())