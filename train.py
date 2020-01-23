import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from CFGAN import CFGAN

def train():
    # Model parameters
    num_epochs = 2000
    d_steps = 128
    g_steps = 128*5
    # action function
    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh
    # gaussian distribution
    # 11 attributes and every attribute need one z
    Attribute_Num = 11

    for epoch in range(num_epochs):
        for 
        fake_data = CFGAN(generator_activation_function, torch.randn(Attribute_Num))

    