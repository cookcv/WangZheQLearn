import torchvision
import torch

from models.my_resnet import myResnet
from models.state_model import StateModel

def get_feature_model(device,pretrained=True,requires_grad=False):
    resnet101 = torchvision.models.resnet101(pretrained).eval().cuda(device).requires_grad_(requires_grad)
    resnet101 = myResnet(resnet101)

    return resnet101

def get_state_model(device,model_weights,requires_grad=True):

    model_get_state=StateModel(6,768,2,12,0.0,6*6*2048)
    model_get_state.load_state_dict(torch.load(model_weights))
    model_get_state.cuda(device).requires_grad_(requires_grad)

    return model_get_state