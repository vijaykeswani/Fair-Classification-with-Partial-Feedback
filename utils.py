import os
import sys
import random
import warnings
import numpy as np
import itertools as iter
from copy import deepcopy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.optimize import minimize 
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import folktables
from folktables import ACSDataSource
import aif360
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import GermanDataset

def adult_preprocess(ACSIncome, features, labels, groups, attr, PROT_GRP_INDEX):
    race_index = ACSIncome.features.index(attr)
    groups = features[:,race_index]
    idx = (groups <= 2)
    groups = groups[idx]
    labels = labels[idx]
    features = features[idx]    
    
    new_groups = np.array([1 if g == 1 else 0 for g in groups])
    features[:,PROT_GRP_INDEX] = np.array([1 if g == 1 else 0 for g in features[:,PROT_GRP_INDEX]])
    
    for i in range(len(features[0])):
        if i == PROT_GRP_INDEX:
            continue
        
        features[:,i] = (features[:,i] - np.mean(features[:,i]))/np.std(features[:,i])
    
    return features, labels, new_groups


def get_adult_dataset(protected_attribute):
    data_source = ACSDataSource(survey_year='2019', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True) ##change to False if data has already been downloaded once

    ACSIncome = folktables.BasicProblem(
        features=[
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'WKHP',
            'RAC1P',
            'SEX',
        ],
        target='PINCP',
        target_transform=lambda x: x > 50000,    
        group='RAC1P',
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    if protected_attribute == "race":
        attr = "RAC1P"
    else:
        attr = "SEX"
        
    features, labels, groups = ACSIncome.df_to_numpy(acs_data)

    PROT_GRP_INDEX = ACSIncome.features.index(attr)
    features, labels, groups = adult_preprocess(ACSIncome, features, labels, groups, attr, PROT_GRP_INDEX)
    dataset_all = (features, labels, groups)
    
    return dataset_all, PROT_GRP_INDEX
    
    
def get_german_dataset():

    label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
    protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]

    gd = GermanDataset(protected_attribute_names=['sex'], privileged_classes=[['male']], 
                       metadata={'label_map': label_map, 'protected_attribute_maps': protected_attribute_maps})

    PROT_GRP_INDEX = gd.feature_names.index('sex')

    features = np.hstack((gd.features[:,:8], gd.labels))
    gd_labels = [1 if l==[1] else 0 for l in gd.labels]

    # df = pd.DataFrame(features, columns=gd.feature_names[:len(features[0])-1] + ["class_label"])
    df = pd.DataFrame(features, columns=gd.feature_names[:8] + ["class_label"])

    df = df.sample(frac=1).reset_index(drop=True)
    df.columns, np.mean(df['sex']), df['class_label'].value_counts()    
    dataset_pd_1 = df.head(500)
    dataset_pd_2 = df.iloc[np.random.randint(0, len(df), size=20000)]
    
    features_1 = np.array(dataset_pd_1)[:, :-1]
    features_2 = np.array(dataset_pd_2)[:, :-1]


#     # sanity check
#     print (np.mean(np.array(dataset_pd_2)[:, -1]), np.mean(gd.labels))

#     for j in range(len(features_2[0])):
#         print (np.mean(features_2[:,j]), np.mean(gd.features[:,j]))

    # preprocessing
    labels_1 = [1 if l==1 else 0 for l in np.array(dataset_pd_1)[:, -1]]
    labels_2 = [1 if l==1 else 0 for l in np.array(dataset_pd_2)[:, -1]]

    for i in range(len(features_1[0])):
        if i == PROT_GRP_INDEX:
            continue

        features_1[:,i] = (features_1[:,i] - np.mean(features_1[:,i]))/np.std(features_1[:,i])
        features_2[:,i] = (features_2[:,i] - np.mean(features_2[:,i]))/np.std(features_2[:,i])    

    features_1 = np.hstack((features_1, np.ones((len(features_1),1))))
    features_2 = np.hstack((features_2, np.ones((len(features_2),1))))

    dataset_all = (features_1, labels_1, features_2, labels_2)
    return dataset_all, PROT_GRP_INDEX
    
def get_stats(pred, data_cur, PROT_GRP_INDEX):
    # write functions to compute the following
    stats = {}
    
    X = data_cur[0]
    
    y = data_cur[1]
    y = np.array([int(a) for a in y])
    
    N = len(y)
    pred = [int(p) for p in pred]
    pred = np.array(pred)
    stats['accuracy'] = sum(pred == y) / N
    
    stats['fdr'] = sum((pred == 1) & (y == 0)) / sum(pred)
    
    # revenue 
    rev = [0, 0, -500, 200] # TODO: Fix appropriate numbers
    stats['revenue']  = rev[0] * sum((pred == 0) & (y == 0))
    stats['revenue'] += rev[1] * sum((pred == 0) & (y == 1))
    stats['revenue'] += rev[2] * sum((pred == 1) & (y == 0))
    stats['revenue'] += rev[3] * sum((pred == 1) & (y == 1))
    
    # number of loans given
    stats['total_loans'] = sum((pred == 1))
    
    
    # group-wise statistics
    if PROT_GRP_INDEX != -1: 
        # dataset does has protected group information
    
        grp = np.array(X)[:, PROT_GRP_INDEX]
        for g in range(2):
            Ng = sum(grp == g)
            if Ng == 0: 
                continue 
            
            # accuracy (groupwise)
            stats[f'accuracy-{g}'] = sum((pred != y) & (grp == g)) / Ng
            
            # FDR (groupwise)
            stats[f'fdr-{g}'] = sum((pred == 1) & (y == 0) & (grp == g))  
            stats[f'fdr-{g}'] /= sum((pred==1) & (grp == g))
            
            # revenue (groupwise) 
            stats[f'revenue-{g}']  = rev[0] * sum((pred == 0) & (y == 0) & (grp == g))
            stats[f'revenue-{g}'] += rev[1] * sum((pred == 0) & (y == 1) & (grp == g))
            stats[f'revenue-{g}'] += rev[2] * sum((pred == 1) & (y == 0) & (grp == g))
            stats[f'revenue-{g}'] += rev[3] * sum((pred == 1) & (y == 1) & (grp == g))
            
            # number of loans given (groupwise)
            stats[f'total_loans-{g}'] = sum((pred == 1) & (grp == g))
            stats[f'frac_loans-{g}'] = sum((pred == 1) & (grp == g))/sum((grp == g))
            
            # fraction of loans approved (groupwise)
            stats[f'stat_rate-{g}'] = sum((pred == 1) & (grp == g)) / Ng
            stats[f'tp-{g}'] = sum((pred == 1) & (grp == g) & (y == 1)) / sum((grp == g) & (y == 1))
    
    stats['stat_rate'] = (stats['stat_rate-0'] - stats['stat_rate-1'])
    stats['tp_rate'] = (stats['tp-0'] - stats['tp-1'])
    
    return stats

def get_avg_stats(data, PROT_GRP_INDEX):
    keys = ['accuracy', 'fdr', 'revenue', 'total_loans', 'stat_rate']
    
    tmp_keys = []
    if PROT_GRP_INDEX != -1:
        # dataset does has protected group information
        for g in range(2):
            for k in keys:
                tmp_keys.append(k + "-" + str(g))
            
    keys.extend(tmp_keys)
#     print (keys)
            
    avg_stats = {}
    for k in keys:
        avg_stats[k] = 0
        
    for k in keys:
        for row in data:
            avg_stats[k] += row[k]
            
        avg_stats[k] /= len(data)
        
    return avg_stats

def beautify(stats, indent = "\t\t"):
    beautiful_string = ""
    for k in stats.keys():
        beautiful_string += f'{indent}{k}:\t{np.round(stats[k], 2)}'
        beautiful_string += '\n'
    
    return beautiful_string



def get_adult_data_splits_and_initial_clf(dataset_all, ITERS):
    features, labels, groups = dataset_all
    X_TRAIN, X_TEST, y_TRAIN, y_TEST, group_TRAIN, group_TEST = train_test_split(
        features, labels, groups, test_size=0.01, shuffle=True)

    X_TRAIN = np.hstack((X_TRAIN, np.ones((len(X_TRAIN),1))))
    X_TEST = np.hstack((X_TEST, np.ones((len(X_TEST),1))))

    DATA_TRAIN = (X_TRAIN, y_TRAIN)
    DATA_TEST = (X_TEST, y_TEST)

    # Ensure that training data's size is a multiple of ITERS
    X_TRAIN = X_TRAIN[:len(X_TRAIN) - len(X_TRAIN)%ITERS] 
    y_TRAIN = y_TRAIN[:len(y_TRAIN) - len(y_TRAIN)%ITERS] 

    # Generate data to use at each iteration
    tmp_X = np.array_split(X_TRAIN, ITERS)
    tmp_y = np.array_split(y_TRAIN, ITERS)

    # Datasets S0, S1, S2, .... used throughout the algorithm 
    data_s = [[X, y] for (X,y) in zip(tmp_X, tmp_y)] 
    
    initial_balance = 0.90  ## fraction of samples in L_0 with label 1

    data_s[0] = [np.array(data_s[0][0]), np.array(data_s[0][1])]    
    indices_1 = np.argwhere(data_s[0][1] == 1)[:,0]
    indices_0 = np.argwhere(data_s[0][1] == 0)[:,0]

    random.shuffle(indices_1)
    random.shuffle(indices_0)

    N_1 = int(initial_balance * len(data_s[0][1])/2)
    N_0 = int((1-initial_balance) * len(data_s[0][1])/2)
    
    data_init, y_init = [[], []], []
    data_init[0] += [data_s[0][0][i] for i in indices_1[:N_1]]
    data_init[1] += [int(data_s[0][1][i]) for i in indices_1[:N_1]]
    data_init[0] += [data_s[0][0][i] for i in indices_0[:N_0]]
    data_init[1] += [int(data_s[0][1][i]) for i in indices_0[:N_0]]
    
    
    X_init = data_init[0] + [data_s[0][0][i] for i in indices_1[N_1:]] + [data_s[0][0][i] for i in indices_0[N_0:]]
    y_init = [1 for _ in range(len(data_init[0]))] + [0 for _ in range(len(X_init) - len(data_init[0]))]
    X_init_2 = [data_s[0][0][i] for i in indices_1[N_1:]] + [data_s[0][0][i] for i in indices_0[N_0:]]
    
    sample_weights = np.ones(len(y_init))
    clf = LogisticRegression(random_state=0, max_iter=10000, class_weight="balanced").fit(X_init, y_init)
    initial_clf_cons = [clf.coef_[0], 0.5]
    
    data_l = [[], []]
    data_l_exp = [[], []]

    for i in range(len(data_init[0])): 
        data_l[0].append(data_init[0][i]) # add X
        data_l[1].append(data_init[1][i]) # add y
        data_l_exp[0].append(data_init[0][i]) # add X
        data_l_exp[1].append(data_init[1][i]) # add y
        
    for i in range(len(X_init_2)): 
        data_l[0].append(X_init_2[i]) # add X
        data_l[1].append(0) # add 0
    
    return data_l, data_s, initial_clf_cons


def get_german_data_splits_and_initial_clf(dataset_all, ITERS):

    features_1, labels_1, features_2, labels_2 = dataset_all
    X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(
        features_2, np.array(labels_2), test_size=0.01, shuffle=True)

    DATA_TRAIN = (X_TRAIN, y_TRAIN)
    DATA_TEST = (X_TEST, y_TEST)

    # Ensure that training data's size is a multiple of ITERS
    X_TRAIN = X_TRAIN[:len(X_TRAIN) - len(X_TRAIN)%ITERS] 
    y_TRAIN = y_TRAIN[:len(y_TRAIN) - len(y_TRAIN)%ITERS] 

    # Generate data to use at each iteration
    tmp_X = np.array_split(X_TRAIN, ITERS)
    tmp_y = np.array_split(y_TRAIN, ITERS)

    # Datasets S0, S1, S2, .... used throughout the algorithm 
    data_s = [[X, y] for (X,y) in zip(tmp_X, tmp_y)] 
    
    initial_balance = 0.9  ## fraction of samples in L_0 with label 1

    data_s[0] = [np.array(data_s[0][0]), np.array(data_s[0][1])]    
    features_1 = data_s[0][0]    
    labels_1 = np.array(data_s[0][1])
    
    indices_1 = np.argwhere(labels_1 == 1)[:,0]
    indices_0 = np.argwhere(labels_1 == 0)[:,0]

    random.shuffle(indices_1)
    random.shuffle(indices_0)

    N_1 = int(initial_balance * len(labels_1))
    N_0 = int((1-initial_balance) * len(labels_1))
    
    data_init, y_init = [[], []], []
    data_init[0] += [features_1[i] for i in indices_1[:N_1]]
    data_init[1] += [int(labels_1[i]) for i in indices_1[:N_1]]
    data_init[0] += [features_1[i] for i in indices_0[:N_0]]
    data_init[1] += [int(labels_1[i]) for i in indices_0[:N_0]]
    
    
    X_init = data_init[0] + [features_1[i] for i in indices_1[N_1:]] + [features_1[i] for i in indices_0[N_0:]]
    y_init = [1 for _ in range(len(data_init[0]))] + [0 for _ in range(len(X_init) - len(data_init[0]))]
    X_init_2 = [features_1[i] for i in indices_1[N_1:]] + [features_1[i] for i in indices_0[N_0:]]
        
    sample_weights = np.ones(len(y_init))
    clf = LogisticRegression(random_state=0, max_iter=10000, class_weight="balanced").fit(X_init, y_init)
    initial_clf_cons = [clf.coef_[0], 0.5]

    data_l = [[], []]
    data_l_exp = [[], []]

    for i in range(len(data_init[0])): 
        data_l[0].append(data_init[0][i]) # add X
        data_l[1].append(data_init[1][i]) # add y
        data_l_exp[0].append(data_init[0][i]) # add X
        data_l_exp[1].append(data_init[1][i]) # add y
        
    for i in range(len(X_init_2)): 
        data_l[0].append(X_init_2[i]) # add X
        data_l[1].append(0) # add 0

    return data_l, data_s, initial_clf_cons
        