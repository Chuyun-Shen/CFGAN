import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from load_data import load_data
from CFGAN import CFGAN, Discriminator


def train():
    # Model parameters
    num_epochs = 1000
    g_steps = 1000
    batch = 128
    LR = 0.001
    # action function
    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh

    # net init
    discriminator_1 = Discriminator(
        discriminator_activation_function, 11, 64, 1)
    generator = CFGAN(generator_activation_function)

    # Binary cross entropy: https://pytorch.org/docs/stable/nn.html?highlight=bceloss#torch.nn.BCELoss
    criterion = nn.BCELoss()
    # optim
    generator_optim = torch.optim.Adam(
        discriminator_1.parameters(), lr=LR, betas=(0.9, 0.99))
    generator_optim = torch.optim.Adam(
        generator.parameters(), lr=LR, betas=(0.9, 0.99))

    dataloader = load_data(batch)
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        # D.zero_grad()
        data = dataloader.copy()

        for real_data in data:

            # 1A: Train D1 on real
            discriminator_1.zero_grad()
            d_real_data = real_data
            # real data's lable should be true
            d_real_labe = torch.ones(batch)
            d_real_decision = discriminator_1(d_real_data)
            d_real_loss = criterion(d_real_decision, d_real_labe)
            d_real_loss.backward()

            # 1B: Train D1 on fake data
            d_fake_data = generator(torch.randn(batch, 11))
            d_fake_lable = torch.ones(batch, 1)
            d_fake_decision = discriminator_1(d_fake_data)
            d_fake_loss = criterion(d_fake_decision, d_fake_lable)
            d_fake_loss.backward()
            # Only optimizes D1's parameters
            discriminator_1.step()

        #  todo   dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
        for g_index in range(g_steps):
            # Train G on D's response
            generator.zero_grad()

            g_fake_data = generator(torch.randn(batch, 11))
            d_g_fake_decision = discriminator_1(g_fake_data)
            g_fake_lable = torch.ones(batch, 1)
            g_error = criterion(d_g_fake_decision, g_fake_lable)
            g_error.backward()
            g_optimizer.step()
        # for g_index in range(g_steps):
        #     # 2. Train G on D's response (but DO NOT train D on these labels)
        #     G.zero_grad()

        #     gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        #     g_fake_data = G(gen_input)
        #     dg_fake_decision = D(preprocess(g_fake_data.t()))
        #     g_error = criterion(dg_fake_decision, Variable(
        #         torch.ones([1, 1])))  # Train G to pretend it's genuine

        #     g_error.backward()
        #     g_optimizer.step()  # Only optimizes G's parameters
        #     ge = extract(g_error)[0]

        # gaussian distribution
        fake_data = CFGAN(generator_activation_function,
                          torch.randn(Attribute_Num))
