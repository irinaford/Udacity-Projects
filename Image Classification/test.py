# References:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html
#   https://sagemaker.readthedocs.io/en/v2.23.1/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models

import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

JPEG_CONTENT_TYPE = 'image/png'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # calling gpu

def Net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
               nn.Linear(num_features, 256),
               nn.ReLU(inplace=True),
               nn.Linear(256, 133))
    return model  


def model_fn(model_dir):
    model = Net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception('Unsupported ContentType in content_type: {}'.format(content_type))


def predict_fn(input_object, model):
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    input_object=test_transform(input_object)
    input_object = input_object.to(device) 
    with torch.no_grad():
        return model(input_object.unsqueeze(0))

