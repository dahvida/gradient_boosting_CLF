"""
Evaluation functions to measure the performance of a given loss function on a 
given dataset
"""

import sys
sys.path.append('..')
from losses.losses import *
from misc.utils import *
from misc.loaders import *

import lightgbm as lgb
from sklearn.metrics import *
from scipy.special import expit
from sklearn.model_selection import train_test_split
from scipy import optimize as opt
from sklearn.preprocessing import binarize
import numpy as np

###############################################################################

def eval_loss_molnet(loss_name, dataset_name, tasks, metric):
    #preallocate results matrix (n_metrics, n_tasks)
    results = np.empty((50, tasks, 8))       
                                                 
    for j in range(tasks):
        #fetch processed dataset
        x, y = molnet_loader(dataset_name, j)     

        #define search space, optimize in CV                                      
        search = create_param_space(loss_name)                                              
        optimum = optimize_CV(x, y, loss_name, search, metric = metric)   
                 
        for i in range(50):
            #get random splits (80:10:10)
            train_x, val_x, train_y, val_y = train_test_split(x, y,
                                                              stratify=y,
                                                              test_size=0.2,
                                                              random_state=i)                
            test_x, val_x, test_y, val_y = train_test_split(val_x, val_y,
                                                            stratify=val_y,
                                                            test_size=0.5,
                                                            random_state=i)      

            #create loss container with opt params and LightGBM datasets            
            objective = create_objective(loss_name, optimum, train_y)                       
            train = lgb.Dataset(train_x, train_y)                                           
            val = lgb.Dataset(val_x, val_y)                       

            #train model with optimal params and early stopping on val                          
            model = train_model(train, val, objective, optimum)                             
            
            #get adjusted predictions and store results
            val_p = expit(model.predict(val_x) + objective.loss_function.init_score)
            test_p = expit(model.predict(test_x) + objective.loss_function.init_score)
            roc, pr, acc, bacc, pre, rec, f1, mcc = get_metrics(val_y, val_p, test_y, test_p)
            results[i, j, 0] = roc
            results[i, j, 1] = pr
            results[i, j, 2] = acc
            results[i, j, 3] = bacc
            results[i, j, 4] = pre
            results[i, j, 5] = rec
            results[i, j, 6] = f1
            results[i, j, 7] = mcc                                   
    
    return results



def eval_loss_moldata(loss_name, dataset_name, tasks):
    #preallocate results matrix (n_replicates, n_tasks, n_metrics)
    results = np.empty((5, tasks, 8))
    
    for j in range(tasks):
        #fetch processed dataset
        train_x, train_y, test_x, test_y, val_x, val_y = moldata_loader(dataset_name, j)
        
        #define search space, optimize in CV 
        search = create_param_space(loss_name)
        optimum = optimize(train_x, train_y,
                           val_x, val_y,
                           loss_name, search)
        
        for i in range(5):
            #create loss container with opt params and LightGBM datasets  
            train = lgb.Dataset(train_x, train_y)
            val = lgb.Dataset(val_x, val_y)
            objective = create_objective(loss_name, optimum, train_y)
            
            #train model with optimal params and early stopping on val                          
            model = train_model(train, val, objective, optimum)
            
            #get adjusted predictions for both val and test
            val_p = expit(model.predict(val_x) + objective.loss_function.init_score)
            test_p = expit(model.predict(test_x) + objective.loss_function.init_score)
            
            #get all metrics and store results
            roc, pr, acc, bacc, pre, rec, f1, mcc = get_metrics(val_y, val_p, test_y, test_p)
            results[i, j, 0] = roc
            results[i, j, 1] = pr
            results[i, j, 2] = acc
            results[i, j, 3] = bacc
            results[i, j, 4] = pre
            results[i, j, 5] = rec
            results[i, j, 6] = f1
            results[i, j, 7] = mcc
            
    return results



def get_metrics(val_y, val_p, test_y, test_p):
    #get best threshold on val set according to MCC
    #this step is necessary since changing the loss function affects
    #the probability distribution of the outputs, so that unlike for weighted
    #cross-entropy T=0.5 does not necessarily yield optimal results
    res = opt.minimize_scalar(
            lambda p: -matthews_corrcoef(val_y, binarize(val_p.reshape(-1,1), threshold=p)),
            bounds=(0, 1),
            method='bounded'
        )
    opt_t = res.x
    
    #binarize probs on test set according to optimal threshold on val set
    test_l = binarize(test_p.reshape(-1,1), threshold=opt_t)
    
    #compute all metrics
    roc = roc_auc_score(test_y, test_p)
    pr = average_precision_score(test_y, test_p)
    acc = accuracy_score(test_y, test_l)
    bacc = balanced_accuracy_score(test_y, test_l)
    pre = precision_score(test_y, test_l)
    rec = recall_score(test_y, test_l)
    f1 = f1_score(test_y, test_l)
    mcc = matthews_corrcoef(test_y, test_l)
    
    return roc, pr, acc, bacc, pre, rec, f1, mcc




