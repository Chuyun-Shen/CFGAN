import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from load_data import dataLoader
from CFGAN import CFGAN, Discriminator

def train():
    # Model parameters
    num_epochs = 1000
    d_steps = 128
    g_steps = 128*5

    # action function
    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh
    
    # Binary cross entropy: https://pytorch.org/docs/stable/nn.html?highlight=bceloss#torch.nn.BCELoss
    criterion = nn.BCELoss()
    
    # optim
      
    

    for d_index in range(d_steps):
            # 1. Train D on real+fake
            # D.zero_grad()
            
            discriminator_1 = Discriminator(discriminator_activation_function, 11, 64, 1)
            for real_data in dataLoader:

                # 1A: Train D on real
                discriminator_1.zero_grad()
                d_real_data = real_data
                # real data's lable should be true
                d_real_labe = torch.ones(d_real_data.size())
                d_real_decision = discriminator_1(d_real_data)
                d_real_loss = criterion(d_real_decision, d_real_labe)
                d_real_loss.backward()
                
                # 1B: Train D1 on fake data  
        #     d_real_data = Variable(d_sampler(d_input_size))
        #     d_real_decision = D(preprocess(d_real_data))
        #     # all real data's lable should be 1
        #     d_real_error = criterion(d_real_decision, Variable(
        #         torch.ones([1, 1])))  # ones = true
        #     d_real_error.backward()  # compute/store gradients, but don't change params

        #     #  1B: Train D on fake
        #     d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        #     # detach to avoid training G on these labels
        #     d_fake_data = G(d_gen_input).detach()
        #     d_fake_decision = D(preprocess(d_fake_data.t()))
        #     d_fake_error = criterion(d_fake_decision, Variable(
        #         torch.zeros([1, 1])))  # zeros = fake
        #     d_fake_error.backward()
        #     # Only optimizes D's parameters; changes based on stored gradients from backward()
        #     d_optimizer.step()

        #     dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

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
        fake_data = CFGAN(generator_activation_function, torch.randn(Attribute_Num))

    