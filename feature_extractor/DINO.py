import torch
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def get_model(model):
    if model == "vits8":
        patch_size = 8
        vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        return vits8, patch_size


def get_DINO_features(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    # print("[inside get_DINO_feature...]: ")
    with torch.no_grad(): #disable gradiente calculation 
        out = model.get_intermediate_layers(frame.unsqueeze(0).cpu(), n=15)[-1] #check after
        out = out[:, 1:, :]  # we discard the [CLS] token
        h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
        dim = out.shape[-1]
        out = out[0].reshape(h, w, dim)
        out = out.reshape(-1, dim)
        if return_h_w:
            return out, h, w
        return out 

if __name__ == "__main__":
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import sys
    import argparse
    import cv2
    import random
    import colorsys
    import requests
    from io import BytesIO

    import skimage.io
    from skimage.measure import find_contours
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms as transforms
    import numpy as np
    from PIL import Image

    
    device = torch.device("cuda")
    model, _ = get_model("vits8")
    model = model.to(device)
    # print(model)