import timm
from torch import nn

def get_model(model_name: str = 'resnest14d'):
    model = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=5)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=5, bias=True),
        nn.Sigmoid()
    )
    return model