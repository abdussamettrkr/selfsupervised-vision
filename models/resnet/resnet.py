from torchvision import  models


def resnet50(**kwargs):
    return models.resnet18(pretrained=False)

def resnet34(**kwargs):
    return models.resnet18(pretrained=False)

def resnet18(**kwargs):
    return models.resnet18(pretrained=False)

