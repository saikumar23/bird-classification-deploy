import torch.nn as nn
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Input: 64 x 3 x 64 x 64
        self.conv1 = conv_block(in_channels, 64) # 64 x 64 x 64 x 64
        self.conv2 = conv_block(64, 128, pool=True) # 64 x 128 x 32 x 32
        self.res1 = nn.Sequential(conv_block(128, 128), # 64 x 128 x 32 x 32
                                  conv_block(128, 128)) # 64 x 128 x 32 x 32
        
        self.conv3 = conv_block(128, 256, pool=True) # 64 x 256 x 16 x 16
        self.conv4 = conv_block(256, 512, pool=True) # 64 x 512 x 8 x 8 
        self.res2 = nn.Sequential(conv_block(512, 512), # 64 x 512 x 8 x 8 
                                  conv_block(512, 512)) # 64 x 512 x 8 x 8 
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), # 64 x 512 x 1 x 1 
                                        nn.Flatten(), # 64 x 512
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out