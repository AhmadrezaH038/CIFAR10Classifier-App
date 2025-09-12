from torch import nn

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters

        # Main Path
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += shortcut
        out = self.relu(out)

        return out



class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=2):
        super(ConvolutionalBlock, self).__init__()
        F1, F2, F3 = filters

        # Main Path
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F3)

        # Shortcut Path
        self.shortcut_conv = nn.Conv2d(in_channels, F3, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(F3)

        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += shortcut
        out = self.relu(out)

        return out
    


class ResNet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet50, self).__init__()

        # Stage 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        # Stage 1
        self.stage1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvolutionalBlock(in_channels=64, filters=[64, 64, 256], stride=1),
            IdentityBlock(in_channels=256, filters=[64, 64, 256]),
            IdentityBlock(in_channels=256, filters=[64, 64, 256])
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvolutionalBlock(in_channels=256, filters=[128, 128, 512], stride=2),
            IdentityBlock(in_channels=512, filters=[128, 128, 512]),
            IdentityBlock(in_channels=512, filters=[128, 128, 512]),
            IdentityBlock(in_channels=512, filters=[128, 128, 512]),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            ConvolutionalBlock(in_channels=512, filters=[256, 256, 1024], stride=2),
            IdentityBlock(in_channels=1024, filters=[256, 256, 1024]),
            IdentityBlock(in_channels=1024, filters=[256, 256, 1024]),
        )


        # Final Layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)
    

    def forward(self, x):
        # stage 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Stages 2-4
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final layers
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

