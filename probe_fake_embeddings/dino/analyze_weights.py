import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import vision_transformer as vits

import visdom as vis
import utils
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        #self.linear.weight.data.normal_(mean=0.0, std=0.01)
        #self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        #print(x)
        #x = x.view(x.size(0), -1)

        # linear layer
        out = self.linear(x)
        out = nn.functional.sigmoid(out)
        return out

embed_dim = 0

ckp_path = "checkpoint.pth.tar"

if "vit_small" in vits.__dict__.keys():
    model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
    embed_dim = model.embed_dim * (4 + int(False))

checkpoint = torch.load(ckp_path, map_location="cpu")

linear_classifier = LinearClassifier(embed_dim, num_labels=1)

if "state_dict" in checkpoint:
    msg = linear_classifier.load_state_dict(checkpoint["state_dict"], strict=False)
    print("=> loaded '{}' from checkpoint '{}' with msg {}".format("state_dict", ckp_path, msg))
        


w, b = linear_classifier.parameters()
w = w.detach().numpy()
w = np.sort(w, axis=-1)
print(w.shape)
print(w[0,-30:])