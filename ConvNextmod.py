import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image

# load the pre-trained ConvNeXT model from the PyTorch Hub
base_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

# get the output from the layer before the prediction layer
output_layer = nn.Sequential(*list(base_model.children())[:-1])
# create a new model using the output layer
feature_model = nn.Sequential(*list(base_model.children())[:-1])

# define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img):
    # load the image
    image = img.convert('RGB')
    # apply the image transform
    image = transform(image)
    # add batch dimension
    image = image.unsqueeze(0)
    # set the model to evaluation mode
    feature_model.eval()
    # get the feature vector for the image
    with torch.no_grad():
        features = feature_model(image)
    # convert the feature vector to a numpy array
    features = features.squeeze().numpy()
    return features
