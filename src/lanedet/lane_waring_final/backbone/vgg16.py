import torch
import torch.nn as nn
import config
import os


class VGG16Encode(nn.Module):
    def __init__(self, batch_norm=True, pretrained=False):
        super(VGG16Encode, self).__init__()
        self.cfg = [[64, 64], ['M', 128, 128], ['M', 256, 256, 256], ['M', 512, 512, 512], ['M', 512, 512, 512]]
        # remove the last maxpooling
        self.batch_norm = batch_norm

        self.layer1 = self.make_layers(cfg=self.cfg[0], in_channels=3)
        self.layer2 = self.make_layers(cfg=self.cfg[1], in_channels=64)
        self.layer3 = self.make_layers(cfg=self.cfg[2], in_channels=128)
        self.layer4 = self.make_layers(cfg=self.cfg[3], in_channels=256)
        self.layer5_seg = self.make_layers(cfg=self.cfg[4], in_channels=512)
        self.layer5_ins = self.make_layers(cfg=self.cfg[4], in_channels=512)
        if not pretrained:
            self._initialize_weights(self.layer1)
            self._initialize_weights(self.layer2)
            self._initialize_weights(self.layer3)
            self._initialize_weights(self.layer4)
            self._initialize_weights(self.layer5_seg)
            self._initialize_weights(self.layer5_ins)
        else:
            self.load_weights()

    def forward(self, x):
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        feature5_seg = self.layer5_seg(feature4)
        feature5_ins = self.layer5_ins(feature4)

        return feature1, feature2, feature3, feature4, feature5_seg, feature5_ins

    def _initialize_weights(self, layers):
        for m in layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def load_weights(self):
        weight_file = config.pretrained_weights['vgg16_bn']
        if os.path.exists(weight_file) is False:
            print("file not exists ", weight_file)
        self.weights = torch.load(weight_file)
        keys = list(self.weights.keys())
        layer1_key = keys[:12]
        layer2_key = keys[12:24]
        layer3_key = keys[24:42]
        layer4_key = keys[42:60]
        layer5_key = keys[60:78]
        self.load_layer(self.layer1, layer1_key)
        self.load_layer(self.layer2, layer2_key)
        self.load_layer(self.layer3, layer3_key)
        self.load_layer(self.layer4, layer4_key)
        self.load_layer(self.layer5_seg, layer5_key)
        self.load_layer(self.layer5_ins, layer5_key)
        print('loaded weights for all feature layers !')

    def load_layer(self, layer, keys):
        i = 0
        for l in layer:
            if isinstance(l, nn.Conv2d):
                l.weight.data.copy_(self.weights[keys[i]])
                i += 1
                l.bias.data.copy_(self.weights[keys[i]])
                i += 1
            if isinstance(l, nn.BatchNorm2d):
                l.weight.data.copy_(self.weights[keys[i]])
                i += 1
                l.bias.data.copy_(self.weights[keys[i]])
                i += 1
                l.running_mean.data.copy_(self.weights[keys[i]])
                i += 1
                l.running_var.data.copy_(self.weights[keys[i]])
                i += 1


if __name__ == "__main__":
    net = VGG16Encode(pretrained=False)
    # weight = torch.load(config.pretrained_weights["vgg16_bn"])
    net.load_weights()
    # print(net.layer1[0].weight.data)
    # print(net.layer1[0].weight.data - weight['features.0.weight'])
    data = torch.zeros([1, 3, 512, 512])
    a = net(data)
    print('OK')

