import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torchvision import models as torchvision_models

import utils 
from dataset.augmentation import *
import models.vit as vits
from models.vit import DINOHead
from losses.loss import DINOLoss
from dataset.imageDataset import ImageDataset
from torchvision.datasets import ImageFolder

import time
import math
import sys
from pathlib import Path
import json
import datetime
import numpy as np


def train(train_loader, net, criterion, otimizer, epoch, device):
    pass

def main():
    global args,config, last_epoch, best_epoch, writer