import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import copy

from load_data import load_data
from CFGAN import CFGAN, Discriminator


def train():
    # Model parameters
    num_epochs = 1000
    g_steps = 64
    g2_steps = 64
    batch = 128
    LR = 0.001
    print_interval = 100
    # action function
    discriminator_activation_function = nn.LeakyReLU(0.2)
    generator_activation_function = torch.tanh

    # net init
    discriminator_1 = Discriminator(
        discriminator_activation_function, 11, 64, 1)
    generator = CFGAN(generator_activation_function)

    # Binary cross entropy: https://pytorch.org/docs/stable/nn.html?highlight=bceloss#torch.nn.BCELoss
    criterion = nn.BCELoss()
    # optim
    generator_optim = torch.optim.Adam(
        generator.parameters(), lr=LR, betas=(0.9, 0.99))
    discriminator_1_optim = torch.optim.Adam(
        discriminator_1.parameters(), lr=LR, betas=(0.9, 0.99))

    print("load data")
    dataloader = load_data(batch)
    print("data loading has finished")

    for epoch in range(num_epochs):
        # GAN1
        # 1. Train D on real+fake
        # D.zero_grad()
        data = copy.copy(dataloader)

        for real_data in data:

            # 1A: Train D1 on real
            discriminator_1.zero_grad()
            d_real_data = real_data
            # real data's lable should be true
            d_real_labe = torch.ones(d_real_data.size()[0])
            d_real_decision = discriminator_1(d_real_data.float())
            d_real_loss = criterion(
                torch.squeeze(d_real_decision), d_real_labe)
            d_real_loss.backward()

            # 1B: Train D1 on fake data
            d_fake_data = generator(torch.randn(batch, 11))
            # print(d_fake_data.size())
            d_fake_lable = torch.zeros(batch)
            d_fake_decision = discriminator_1(d_fake_data)
            d_fake_loss = criterion(torch.squeeze(
                d_fake_decision), d_fake_lable)
            d_fake_loss.backward()
            # Only optimizes D1's parameters
            discriminator_1_optim.step()

            drl, dfl = d_real_loss.tolist(), d_fake_loss.tolist()
        for g_index in range(g_steps):
            # Train G on D's response
            generator.zero_grad()

            g_fake_data = generator(torch.randn(batch, 11))
            d_g_fake_decision = discriminator_1(g_fake_data)
            g_fake_lable = torch.ones(batch)
            g_loss = criterion(torch.squeeze(d_g_fake_decision), g_fake_lable)
            g_loss.backward()
            generator_optim.step()
            gl = g_loss.tolist()

        # GAN2
        for g_index in range(g2_steps):
            generator.zero_grad()
            noise_z = torch.randn(batch, 11)
            fake_data = generator(noise_z)
            # O = {race; native country}:(0,0) (0,1) (1,0) (1,1)
            noise_o0 = []
            noise_o1 = []
            noise_o2 = []
            noise_o3 = []
            ge0, ge1, ge2, ge3 = None, None, None, None
            for index, single_data in enumerate(fake_data):
                if(single_data[7] < 0.5 and single_data[9] < 0.5):
                    noise_o0.append(noise_z[index].view(1, -1))
                elif(single_data[7] < 0.5 and single_data[9] >= 0.5):
                    noise_o1.append(noise_z[index].view(1, -1))
                elif(single_data[7] >= 0.5 and single_data[9] < 0.5):
                    noise_o2.append(noise_z[index].view(1, -1))
                else:
                    noise_o3.append(noise_z[index].view(1, -1))
            if(len(noise_o0) != 0):
                noise_o0 = torch.cat(noise_o0)
                o0_0_lable = generator(noise_o0, 0)[:, -1]
                o0_1_lable = generator(noise_o0, 1)[:, -1].detach()
                g_error0 = criterion(o0_0_lable, o0_1_lable)
                g_error0.backward()
                ge0 = g_error0.tolist()
            if(len(noise_o1) != 0):
                noise_o1 = torch.cat(noise_o1)
                o1_0_lable = generator(noise_o1, 0)[:, -1]
                o1_1_lable = generator(noise_o1, 1)[:, -1].detach()
                g_error1 = criterion(o1_0_lable, o1_1_lable)
                g_error1.backward()
                ge1 = g_error1.tolist()
            if(len(noise_o2) != 0):
                noise_o2 = torch.cat(noise_o2)
                o2_0_lable = generator(noise_o2, 0)[:, -1]
                o2_1_lable = generator(noise_o2, 1)[:, -1].detach()
                g_error2 = criterion(o2_0_lable, o2_1_lable)
                g_error2.backward()
                ge2 = g_error2.tolist()
            if(len(noise_o3) != 0):
                noise_o3 = torch.cat(noise_o3)
                o3_0_lable = generator(noise_o3, 0)[:, -1]
                o3_1_lable = generator(noise_o3, 1)[:, -1].detach()
                g_error3 = criterion(o3_0_lable, o3_1_lable)
                g_error3.backward()
                ge3 = g_error3.tolist()

            generator_optim.step()

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G_l (%s err) G_0l (%s) G_1l (%s) G_2l (%s) G_3l (%s);" % (
                epoch, drl, dfl, gl, ge0, ge1, ge2, ge3))


train()
