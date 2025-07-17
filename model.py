import torch.nn as nn
import torchvision.models as models

class ModifiedMobileNetV2(nn.Module):
    def __init__(self):
        super(ModifiedMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)