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
    def __init__(self, f, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size*2)
        self.map3 = nn.Linear(hidden_size*2, output_size)
        # f is action function
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return torch.sigmoid(x)

# a basic Discriminator


class Discriminator(nn.Module):
    def __init__(self, f, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        # f is action function
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))


class CFGAN(nn.Module):
    def __init__(self, f):
        super(CFGAN, self).__init__()

        self.age_net = Generator(
            f, 1, 4, 1)
        self.workclass_net = Generator(
            f, 5, 16, 1)
        self.edu_level_net = Generator(
            f, 6, 16, 1)
        self.marital_status_net = Generator(
            f, 5, 16, 1)
        self.occupation_net = Generator(
            f, 6, 16, 1)
        self.relationship_net = Generator(
            f, 6, 16, 1)
        self.race_net = Generator(
            f, 1, 4, 1)
        self.sex_net = Generator(
            f, 1, 4, 1)
        self.hours_per_week_net = Generator(
            f, 7, 32, 1)
        self.native_country_net = Generator(
            f, 1, 4, 1)
        self.income_net = Generator(
            f, 11, 32, 1)

    def forward(self, input, intervention=-1):
        name = ["race", "age", "sex", "native_country", "marital_status",
                "edu_level", "occupation", "hours_per_week", "workclass", "relationship", "income"]
        Z = dict(zip(name, input.transpose(0, 1).view(len(name), -1, 1)))

        # hight = 0 in the graph
        # sex should considered about intervention
        if(intervention == -1):
            self.sex = self.race_net(Z["sex"])

        elif(intervention == 0):
            self.sex = torch.zeros(Z["sex"].size())
        else:
            self.sex = torch.ones(Z["sex"].size())
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

        return self.age, self.workclass, self.edu_level, self.marital_status,
        self.occupation, self.relationship, self.race, self.sex,
        self.hours_per_week, self.native_country, self.income


cf = CFGAN(f=nn.Sigmoid())
cf(torch.randn(6, 11), intervention=0)
print(cf)
