import os
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms as tfs
import torchvision.models as models
from django.conf import settings
from .threshold import *
from .models import RecentAction
from .resnet50 import ResNet50


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            # Convolutional Neural Network
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),


            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits
    

class ModelManager:
    def __init__(self):

        # simple cnn
        self.device = torch.device('cpu')
        self.simple = SimpleNeuralNetwork().to(self.device)
        simple_path = os.path.join('SimpleNetParameters.pth') #TODO: directory of .pth file
        self.simple.load_state_dict(torch.load(simple_path, weights_only=True))
        self.simple.eval()

        # ResNet NN
        self.resnet = ResNet50(num_classes=10).to(self.device)
        state_dict = torch.load("resnet50.pth", map_location=self.device, weights_only=True)
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        self.resnet.load_state_dict(state_dict, strict=False)  # strict=False ignores mismatches
        self.resnet.eval()
    

        self.transform = tfs.Compose([
            tfs.Resize((32, 32)),
            tfs.ToTensor(),
            tfs.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.2435, 0.2616)),
        ])

        self.labels = [
            'airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck'
        ]

    def classify(self, pil_image):
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out1 = torch.softmax(self.simple(x), dim=1)
            conf1, pred1 = out1.max(1)

            out2 = torch.softmax(self.resnet(x), dim=1)
            conf2, pred2 = out2.max(1)
        
        name, idx, conf = select_model([
            ("SimpleCNN", pred1.item(), conf1.item()),
            ("ResNet",   pred2.item(), conf2.item()),
        ])

        label = self.labels[idx]
        warning = (conf < 0.60)
        return name, label, conf, warning
    

_manager =  ModelManager()


def classify_image(user, image_file):

    img = Image.open(image_file).convert('RGB')
    model_name, label, conf, warning = _manager.classify(img)

    # Log the action
    RecentAction.objects.create(
        user=user,
        model_name=model_name,
        predicted_label=label,
        confidence=conf
    )

    return {
        'model':      model_name,
        'label':      label,
        'confidence': conf,
        'warning':    warning
    }