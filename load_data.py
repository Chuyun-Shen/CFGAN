import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset, DataLoader

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# dataset
class AdultDataset(Dataset):
    def __init__(self, data_set):
        self.x = data_set
        self.len = data_set.size()[0]
    def __getitem__(self,index):
        return self.x[index]
    def __len__(self):
        return self.len

# binary the value

def mapping(tuple):
    # age, 37
    tuple['age'] = 1 if tuple['age'] > 37 else 0
    # workclass
    tuple['workclass'] = 0 if tuple['workclass'] != 'Private' else 1
    # edu-level
    tuple['edu_level'] = 1 if tuple['edu_level'] > 9 else 0
    # maritial statue
    tuple['marital_status'] = 1 if tuple['marital_status'] == "Married-civ-spouse" else 0
    # occupation
    tuple['occupation'] = 1 if tuple['occupation'] == "Craft-repair" else 0
    # relationship
    tuple['relationship'] = 0 if tuple['relationship'] == "Not-in-family" else 1
    # race
    tuple['race'] = 0 if tuple['race'] != "White" else 1
    # sex
    tuple['sex'] = 0 if tuple['sex'] != "Male" else 1
    # hours per week
    tuple['hours_per_week'] = 1 if tuple['hours_per_week'] > 40 else 0
    # native country
    tuple['native_country'] = 1 if tuple['native_country'] == "United-States" else 0
    # income
    tuple['income'] = 1 if tuple['income'] == '>50K' or tuple['income'] == '>50K.' else 0
    return tuple


def load_data(){
    # load data from dataset
    # Using 11variables, the casual model is shown in the paper
    # The atribute are:
    # 0 age: continuous.
    # 1 workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    # 2 (not use)fnlwgt: continuous.
    # 3 (not use)education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    # 4 education-num: continuous.
    # 5 marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    # 6 occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    # 7 relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    # 8 race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    # 9 sex: Female, Male.
    # 10 (not use)capital-gain: continuous.
    # 11 (not use)capital-loss: continuous.
    # 12 hours-per-week: continuous.
    # 13 native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    # 14 output is income: >50K, <=50K


    COLUMNS = ["age", "workclass", "edu_level",
               "marital_status", "occupation", "relationship",
               "race", "sex", "hours_per_week",
               "native_country", "income"]

    train_df = pd.read_csv(
        filepath_or_buffer="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=COLUMNS,
        engine='python',
        usecols=[0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 14],
        sep=r'\s*,\s*',
        na_values="?"
    )

    test_df = pd.read_csv(
        filepath_or_buffer="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        names=COLUMNS,
        skiprows=[0],
        engine='python',
        usecols=[0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 14],
        sep=r'\s*,\s*',
        na_values="?"
    )

    # Drop rows with missing values
    train_df = train_df.dropna(how="any", axis=0)
    test_df = test_df.dropna(how="any", axis=0)

    # To reduce the complexity, we binarize the attribute
    # To reduce the complexity, we binarize the attribute
    train_df = train_df.apply(mapping, axis=1)
    test_df = test_df.apply(mapping, axis=1)

    train_data = torch.from_numpy(train_df.values)
    test_data = torch.from_numpy(test_df.values)
    # merge the datasets
    dataset = torch.cat((train_data, test_data), 0)
    
    adultDataset = AdultDataset(dataset)
    dataLoader = DataLoader(dataset=adultDataset, batch_size=128, shuffle=true)
    return dataLoader
}
