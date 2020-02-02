import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Attribute:
# "age", "workclass", "edu_level","marital_status",
# "occupation", "relationship","race", "sex",
# "hours_per_week","native_country", "income"


# a basic Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size=8):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(input_size, hidden_size, normalize=False),
            *block(hidden_size, hidden_size),
            *block(hidden_size, hidden_size),
            *block(hidden_size, hidden_size),
            *block(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        return x


# a basic Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class CFGAN(nn.Module):
    def __init__(self, z_size):
        super(CFGAN, self).__init__()
        
        self.age_net = Generator(z_size)
        self.workclass_net = Generator(z_size+4)
        self.edu_level_net = Generator(z_size+5)
        self.marital_status_net = Generator(z_size+4)
        self.occupation_net = Generator(z_size+5)
        self.relationship_net = Generator(z_size+5)
        self.race_net = Generator(z_size)
        self.sex_net = Generator(z_size)
        self.hours_per_week_net = Generator(z_size+6)
        self.native_country_net = Generator(z_size)
        self.income_net = Generator(z_size+10)

    def forward(self, input, intervention=-1):
        name = ["race", "age", "sex", "native_country", "marital_status",
                "edu_level", "occupation", "hours_per_week", "workclass", "relationship", "income"]
        Z = dict(zip(name, input.transpose(0, 1)))

        # hight = 0 in the graph
        # sex should considered about intervention
        if(intervention == -1):
            self.sex = self.race_net(Z["sex"])
        elif(intervention == 0):
            self.sex = torch.zeros(Z["sex"].size()[0])
        else:
            self.sex = torch.ones(Z["sex"].size()[0])
        self.age = self.age_net(Z["age"])
        self.race = self.sex_net(Z["race"])
        self.native_country = self.native_country_net(Z["native_country"])

        # hight = 1 in the graph
        self.marital_status = self.marital_status_net(torch.cat(
            [Z["marital_status"], self.race, self.age,
                self.sex, self.native_country], 1
        ))

        # hight = 2 in the gragh
        self.edu_level = self.edu_level_net(torch.cat(
            [Z["edu_level"], self.race, self.age, self.sex,
                self.native_country, self.marital_status], 1
        ))

        # hight = 3 in the gragh
        self.occupation = self.occupation_net(torch.cat(
            [Z["occupation"], self.race, self.age, self.sex,
                self.marital_status, self.edu_level], 1
        ))

        self.hours_per_week = self.hours_per_week_net(torch.cat(
            [Z["hours_per_week"], self.race, self.age, self.sex,
             self.native_country, self.marital_status, self.edu_level], 1
        ))

        self.workclass = self.workclass_net(torch.cat(
            [Z["workclass"], self.age, self.marital_status,
                self.edu_level, self.native_country], 1
        ))

        self.relationship = self.relationship_net(torch.cat(
            [Z["relationship"], self.age, self.sex, self.native_country,
                self.marital_status, self.edu_level], 1
        ))

        # hight = 4 in the gragh

        self.income = self.income_net(torch.cat(
            [Z["income"], self.race, self.age, self.sex, self.native_country, self.marital_status,
                self.edu_level, self.occupation, self.hours_per_week, self.workclass, self.relationship], 1
        ))

        return torch.cat([self.age, self.workclass, self.edu_level, self.marital_status,
                          self.occupation, self.relationship, self.race, self.sex,
                          self.hours_per_week, self.native_country, self.income], 1)



