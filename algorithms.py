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


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


class Algorithm(ABC):
    @abstractmethod
    def __init__(self, empirical_risk_minimizer, data_s, data_u):
        
        # A class that has a fit method
        #     The fit method outputs a classifier given labeled data as input 
        self._empirical_risk_minimizer = empirical_risk_minimizer

        self._data_l = data_l # initial labeled dataset (aka S_0)
        self._data_u = data_u # initial unlabeled dataset (aka U_0)
        
        self._clf_f = self.fit_clf_f(data_l) # classifier to make predictions on _data_l

def get_most_accurate_classifier(data_train):
    X, y = data_train
    
#     clf = LogisticRegression(random_state=0, max_iter=10000, class_weight="balanced").fit(X, y)
    clf = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
    return clf

def update_data_include(pred, data_cur, data_u, data_l):
    for i in range(len(pred)):
        if pred[i] == 1:
            data_l[0].append(data_cur[0][i]) # add X
            data_l[1].append(data_cur[1][i]) # add (true) y
        else:
            data_u[0].append(data_cur[0][i]) # add X
            data_u[1].append(0) # add y
            # no y to be added 
    return (data_l), (data_u)

def get_fair_constrained_classifier(data_train, sample_weights, eps, PROT_GRP_INDEX, C=0):
#     print ("----", PROT_GRP_INDEX)
    X_train, y_train = data_train
    sample_weights = np.array(sample_weights)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    def _log_logistic(X):
        if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
        out = np.empty_like(X)

        idx = X>0
        out[idx] = -np.log(1.0 + np.exp(-X[idx]))
        out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
        return out

    def loss(u):
        N = len(X_train)
        dotp = np.dot(X_train, u)
        
        logloss = - np.dot(sample_weights*y_train, _log_logistic(dotp)) - np.dot(sample_weights*(1-y_train), _log_logistic(1-dotp))
        obj = logloss/N + C * sum(np.power(u, 2))/N
        
#         print (u)
        return obj

    def der(u):
        N = len(X_train)        
        der = np.zeros(d)
        product = (sigmoid(np.dot(X_train, u)) - y_train) * sample_weights
        der = np.dot(X_train.T, product)/N + C * sum(u)/(2*N)
        return der

    # fdr and stat rateconstraint
    def cons(u):
        product = sigmoid(np.dot(X_train, u))
        pred = np.array(product > 0.5)
        
        t_0 = sum(pred & ~ np.array(y_train))/len(y_train)
        t_1 = sum(pred)/len(y_train)
    
        idx = X_train[:,PROT_GRP_INDEX] == 0
        sr_min = np.mean(pred[idx])
        sr_maj = np.mean(pred[~idx])    
    
        cond = [-t_0 + eps*t_1, sr_min - sr_maj + 0.05]        
        return cond
        
    def cons_jac(u):
        product = sigmoid(np.dot(X_train, u))
        jacobian = (X_train.T * (product * (1- product))).T
        
        t_0 = sum([jacobian[i] for i in range(len(y_train)) if y_train[i] == 0])/len(y_train)
        t_1 = sum([jacobian[i] for i in range(len(y_train))])/len(y_train)
        
        jacobian_2 = (X_train.T * (product * (1- product))).T

        idx = X_train[:,PROT_GRP_INDEX] == 0
        sr_min = np.mean(jacobian_2[idx], axis=0)
        sr_maj = np.mean(jacobian_2[~idx], axis=0)    

        cond = [-t_0 + eps*t_1, sr_min - sr_maj]
        return cond

    
    c = 0
    d = len(X_train[0])
    u0 = np.random.rand(d)
    ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons(x), 'jac' : lambda x: cons_jac(x)}

    res = minimize(fun = loss, x0 = u0, method='SLSQP', jac = der, constraints= [ineq_cons],
                   options = {'maxiter': 100, 'ftol': 1e-3, 'eps' : 1e-3, 'disp': False})
    return res.x
    
    
def clf_cons_predict(X, clf_wts):
    product = sigmoid(np.dot(X, clf_wts))
    pred = np.array(product > 0.5)
    return pred
    
def clf_cons_predict_prob(X, clf_wts):
    product = sigmoid(np.dot(X, clf_wts))
    return product
    
    
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx)) 

def get_constrained_classifier(data_train, sample_weights, eps, C=0):
    X_train, y_train = data_train
    sample_weights = np.array(sample_weights)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    def _log_logistic(X):
        if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
        out = np.empty_like(X)

        idx = X>0
        out[idx] = -np.log(1.0 + np.exp(-X[idx]))
        out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
        return out

    def loss(u):
        N = len(X_train)
        dotp = np.dot(X_train, u)
        
        logloss = - np.dot(sample_weights*y_train, _log_logistic(dotp)) - np.dot(sample_weights*(1-y_train), _log_logistic(1-dotp))
        obj = logloss/N + C * sum(np.power(u, 2))/N
        
#         print (u)
        return obj

    def der(u):
        N = len(X_train)        
        der = np.zeros(d)
        product = (sigmoid(np.dot(X_train, u)) - y_train) * sample_weights
        der = np.dot(X_train.T, product)/N + C * sum(u)/(2*N)
        return der

    # fdr constraint
    def cons_fdr(u):
        product = sigmoid(np.dot(X_train, u))
        pred = np.array(product > 0.5)
        
        t_0 = sum(pred & ~ np.array(y_train))/len(y_train)
        t_1 = sum(pred)/len(y_train)
        
        cond = [-t_0 + eps*t_1]
        
        return cond
        
    def cons_jac(u):
        product = sigmoid(np.dot(X_train, u))
        jacobian = (X_train.T * (product * (1- product))).T
        
        t_0 = sum([jacobian[i] for i in range(len(y_train)) if y_train[i] == 0])/len(y_train)
        t_1 = sum([jacobian[i] for i in range(len(y_train))])/len(y_train)
        
        cond = [-t_0 + eps*t_1]
        return cond

    c = 0
    d = len(X_train[0])
    u0 = np.random.rand(d)
    ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_fdr(x), 'jac' : lambda x: cons_jac(x)}

    res = minimize(fun = loss, x0 = u0, method='SLSQP', jac = der, constraints= [ineq_cons],
                   options = {'maxiter': 100, 'ftol': 1e-3, 'eps' : 1e-3, 'disp': False})
    return res.x
    
    
def clf_cons_predict(X, clf_wts):
    product = sigmoid(np.dot(X, clf_wts))
    pred = np.array(product > 0.5)
    return pred
    
def clf_cons_predict_prob(X, clf_wts):
    product = sigmoid(np.dot(X, clf_wts))
    return product
    
    

def get_rev(preds, y):
    c_tp = 2
    c_fp = -5

    idx = preds==True
    rev = c_tp * sum(y[idx] == 1) + c_fp * sum(y[idx] == 0)
#     print (sum(y[idx] == 1), sum(y[idx] == 0))

    return rev

def get_rev_max_classifier(data_train, sample_weights, eps, exploit_fair, PROT_GRP_INDEX, C=0):
    X_train, y_train = data_train
    sample_weights = np.array(sample_weights)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, random_state=10, shuffle=True)
    if exploit_fair:
        clf = get_fair_constrained_classifier([X1, y1], np.ones(len(y1)), eps, PROT_GRP_INDEX, C)
    else:
        clf = get_constrained_classifier([X1, y1], np.ones(len(y1)), eps, C)
    

    probs = sigmoid(np.dot(X2, clf))
    max_rev, opt_threshold = -1000000000, 0.5
    
    for threshold in np.linspace(0, 1, 25):
        preds = np.array(probs > threshold)
        rev = get_rev(preds, y2)
        if rev > max_rev:
            max_rev = rev
            opt_threshold = threshold
            
#         print (threshold, rev)
        
    return [clf, opt_threshold]
    
    
def clf_cons_predict(X, clf_wts, threshold=0.5):
    product = sigmoid(np.dot(X, clf_wts))
    pred = np.array(product > 0.5)
    return pred
    
def clf_cons_predict_prob(X, clf_wts):
    product = sigmoid(np.dot(X, clf_wts))
    return product
    
    
class No_Exploration_Algorithm(Algorithm):
    def __init__(self, eps, data_l, clf_f):
        self._name = "No Exploration Algorithm"
        
        self._clf_f = clf_f

        # initial labeled dataset (aka S_0) and unlabeled dataset (aka U_0) respectively
        self._data_l = [list(data_l[0]), list(data_l[1])]
        self._sample_wt_l = np.ones(len(data_l[1]))
        
        self._eps = eps # fdr constraint
#         self._update_clf_f() # classifier to make predictions on _data_l

    
    def _fit_clf_f(self):
        return get_constrained_classifier(self._data_l, self._sample_wt_l, self._eps, C=1)
        
    def _update_clf_f(self):
        self._clf_f = self._fit_clf_f()
        return 

    def _update_datasets(self, pred, data_cur, weights):
        for i in range(len(pred)):
            if pred[i] == 1:
                self._data_l[0].append(data_cur[0][i]) # add X
                self._data_l[1].append(data_cur[1][i]) # add (true) y
                self._sample_wt_l = np.append(self._sample_wt_l, 1)
        return
        
    def _predict(self, X, eps):
        self._update_clf_f()
        
#         print (self._clf_f)
        clf_preds = clf_cons_predict(X, self._clf_f)
        return clf_preds, [], 0
    

class No_Exploration_Fair_Algorithm(Algorithm):
    def __init__(self, eps, data_l, clf_f):
        self._name = "No Exploration Fair Algorithm"
        
        self._clf_f = clf_f

        # initial labeled dataset (aka S_0) and unlabeled dataset (aka U_0) respectively
        self._data_l = [list(data_l[0]), list(data_l[1])]
        self._sample_wt_l = np.ones(len(data_l[1]))
        
        self._eps = eps # fdr constraint

    
    def _fit_clf_f(self, PROT_GRP_INDEX):
        return get_rev_max_classifier(self._data_l, self._sample_wt_l, self._eps, True, PROT_GRP_INDEX, C=1)
        
    def _update_clf_f(self, PROT_GRP_INDEX):
        self._clf_f = self._fit_clf_f(PROT_GRP_INDEX)
        return 

    def _update_datasets(self, pred, data_cur, weights):
        for i in range(len(pred)):
            if pred[i] == 1:
                self._data_l[0].append(data_cur[0][i]) # add X
                self._data_l[1].append(data_cur[1][i]) # add (true) y
                self._sample_wt_l = np.append(self._sample_wt_l, 1)
        return
        
    def _predict(self, X, eps, PROT_GRP_INDEX):
        self._update_clf_f(PROT_GRP_INDEX)
        
#         print (self._clf_f)
        clf_preds = clf_cons_predict(X, self._clf_f[0], self._clf_f[1])
        return clf_preds, [], 0
    
    
class Target(Algorithm):
    def __init__(self, eps, data_l, clf_f):
        self._name = "Target"
        self._clf_f = clf_f
            
    def _update_datasets(self, pred, data_cur, weights):
        return
        
    def _predict(self, X, eps, PROT_GRP_INDEX):
        clf_preds = clf_cons_predict(X, self._clf_f[0], self._clf_f[1])
        clf_preds = np.array([int(p) for p in clf_preds])
        
        return clf_preds, [], 0
            
        
class KilbertusPaper(Algorithm):
    def __init__(self, eps, data_l, clf_f):
        self._name = "Kilbertus et al. Algorithm"
        
        self._clf_f = clf_f
        self._F = [self._clf_f]
        
        self._data_l = [list(data_l[0]), list(data_l[1])]
        self._sample_wt_l = np.ones(len(data_l[1]))
        
        self._eps = eps # fdr constraint
#         self._update_clf_f() # classifier to make predictions on _data_l

    
    def _update_clf_f(self):
        B = 128
        M = B
        
        new_clf = self._clf_f
#         print (new_clf)
        
        for _ in range(M):  
            indices = np.random.choice(range(len(self._data_l[1])), size=B)
            
            X = np.array([self._data_l[0][i] for i in indices])
            y = [self._data_l[1][i] for i in indices]
            
            clf_probs_new = clf_cons_predict_prob(X, new_clf)
            clf_probs_old = clf_cons_predict_prob(X, self._clf_f)

            grad_u, b0, b1, grad_b0, grad_b1, n0, n1 = 0, 0, 0, 0, 0, 0, 0
            for i in range(len(y)):
                if clf_probs_new[i] > 0.5:
                    grad_u += (y[i] - 0.6) * (1 - clf_probs_new[i])/clf_probs_old[i] * X[i]
                    g = (1 - clf_probs_new[i])/clf_probs_old[i] * X[i]
                    
                    if X[:,PROT_GRP_INDEX][i] == 0:
                        b0 += 1
                        grad_b0 += g
                    else:
                        b1 += 1
                        grad_b1 += g
                        
            grad = grad_u/(b0 + b1) - 0.5 * (b0 - b1)/(b0 + b1) * (grad_b0 - grad_b1)/(b0 + b1)
#                 print (grad)
    
            new_clf += 0.01 * np.mean(grad)
        
#             print ((grad))
        self._F.append(list(new_clf))
        self._clf_f = list(new_clf)
        
        return 

    def _update_datasets(self, pred, data_cur, weights):
        self._data_l = [[], []]
        for i in range(len(pred)):
            if pred[i] == 1:
                self._data_l[0].append(data_cur[0][i]) # add X
                self._data_l[1].append(data_cur[1][i]) # add (true) y
                self._sample_wt_l = np.append(self._sample_wt_l, 1)
        return
        
    def _predict(self, X, eps):
        self._update_clf_f()
        
#         print (self._clf_f)
        clf_preds = clf_cons_predict(X, self._clf_f)
        return clf_preds, [], 0
        

import copy

### Our algorithm
class Exp_Exp_Algorithm(Algorithm):
    def __init__(self, explore_eps, exploit_eps, data_l, clf_f):
        self._name = "Explore Exploit Algorithm"
        
        self._clf_f = clf_f
        self._F = [copy.deepcopy(clf_f)]

        # initial labeled dataset (aka S_0) and unlabeled dataset (aka U_0) respectively
        self._data_l = [list(data_l[0]), list(data_l[1])]
        self._sample_wt_l = np.ones(len(data_l[1]))
        
        self._exploit_eps = exploit_eps # fdr constraint
        self._explore_eps = explore_eps # fdr constraint

    
    def _fit_clf_f(self, eps, exploit_fair, PROT_GRP_INDEX):
        return get_rev_max_classifier(self._data_l, self._sample_wt_l, eps, exploit_fair, PROT_GRP_INDEX, C=1)
        

    def _update_clf_f(self, eps, exploit_fair, PROT_GRP_INDEX):
        self._clf_f = self._fit_clf_f(eps, exploit_fair, PROT_GRP_INDEX)
        self._F.append(copy.deepcopy(self._clf_f))
        return 

    def _update_datasets(self, pred, data_cur, sample_weights):
        for i in range(len(pred)):
            if pred[i] == 1:
                self._data_l[0].append(data_cur[0][i]) # add X
                self._data_l[1].append(int(data_cur[1][i])) # add (true) y
                self._sample_wt_l = np.append(self._sample_wt_l, sample_weights[i])
        return
    
    def _get_weights(self, X):
        F = self._F
        curr_clf = F[-1]
        p_x = np.mean([clf_cons_predict_prob(X, clf) for clf, _ in F[:-1]], axis=0)
        weights = p_x

        return weights
        
        
    def _predict(self, X, y, PROT_GRP_INDEX, exploit_eps, explore_eps, exploit_fair, explore_fair):
        self._update_clf_f(exploit_eps, exploit_fair, PROT_GRP_INDEX)
        
        weights = self._get_weights(X)
        
        clf_preds = clf_cons_predict(X, self._clf_f[0], self._clf_f[1])
        clf_probs = clf_cons_predict_prob(X, self._clf_f[0])
        
        final_preds = [0 for x in X]
        
        prob, ys1, ys2 = [], [], []
        for i, x in enumerate(X):
            if weights[i] > 0.5:   # exploit
                final_preds[i] = int(clf_preds[i])
                prob.append(0)
                ys1.append(y[i])
            else:
                prob.append(clf_probs[i])
                ys2.append(y[i])
    
                
#         print ("Exploited", sum(final_preds), exploit_eps)
        prob = np.array(prob)/sum(prob)
        
        n_exp = int(explore_eps * sum(final_preds))
#         print ("Explored", n_exp, explore_eps)
        
        if explore_fair: 
            gr_0 = np.mean([X[:,PROT_GRP_INDEX][i]==0 and weights[i] < 0.5 for i in range(len(X))])
            gr_1 = np.mean([X[:,PROT_GRP_INDEX][i]==1 and weights[i] < 0.5 for i in range(len(X))])
            
            n_exp_0 =  int(n_exp * gr_0/(gr_0 + gr_1))
            n_exp_1 =  int(n_exp * gr_1/(gr_0 + gr_1))
#             print ("Fair", n_exp_0, n_exp_1)
            
            indices_0 = [i for i in range(len(X)) if X[:,PROT_GRP_INDEX][i]==0]
            prob_0 = [p for i, p in enumerate(prob) if X[:,PROT_GRP_INDEX][i]==0]
            prob_0 = np.array(prob_0)/sum(prob_0)
            
#             print (len(prob_0), len(indices_0))
            explore_indices_0 = np.random.choice(indices_0, size=n_exp_0, p=prob_0, replace=False)
            for i in explore_indices_0:
                final_preds[i] = 1
            
            indices_1 = [i for i in range(len(X)) if X[:,PROT_GRP_INDEX][i]==1]
            prob_1 = [p for i, p in enumerate(prob) if X[:,PROT_GRP_INDEX][i]==1]
            prob_1 = np.array(prob_1)/sum(prob_1)
            explore_indices_1 = np.random.choice(indices_1, size=n_exp_1, p=prob_1, replace=False)
            for i in explore_indices_1:
                final_preds[i] = 1
                
            explore_indices = list(explore_indices_0) + list(explore_indices_1)
            
        else:
            indices = range(len(X))
            explore_indices = np.random.choice(indices, size=n_exp, p=prob, replace=False)
            for i in explore_indices:
                final_preds[i] = 1
            
            
        new_weights = [1 for i in range(len(X))]
                
        
        return final_preds, new_weights, explore_indices
        