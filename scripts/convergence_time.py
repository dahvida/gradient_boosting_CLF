import sys
sys.path.append('..')
from misc.loaders import molnet_loader
from misc.utils import *

import lightgbm as lgb
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import time
import pandas as pd

###############################################################################

def measure_time(x, y, optimum, objective, iters = 5):
    times = []
    rounds = []    
    early_stopping = lgb.early_stopping(stopping_rounds=50, verbose=False)                  #create early stopping callback
    for i in range(iters):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)  #split, create lgbm datasets
        train = lgb.Dataset(x_train, y_train.reshape(-1,1))
        val = lgb.Dataset(x_val, y_val.reshape(-1,1))
        evals_result = {}                                                                   #preallocate dict to store results
        start = time.time()                                                                 #start time
        if objective != None:
            model = lgb.train(params ={                                                     #create model with best params
                            'num_leaves':           int(optimum["num_leaves"]),
                            'learning_rate':        optimum["learning_rate"],
                            'max_depth':            int(optimum["max_depth"]),
                            "max_bin":              int(optimum["max_bin"]),
                            'min_data_in_leaf':     int(optimum["min_data_in_leaf"]),
                            'lambda_l1':            optimum["lambda_l1"],
                            'lambda_l2':            optimum["lambda_l2"],
                            "feature_fraction":     optimum["feature_fraction"],
                            "verbosity":            -100,
                            "seed":                 np.random.randint(0, 100)
                        },
                      train_set = train,
                      fobj=objective.get_gradient,                                          #load custom loss function options
                      feval=objective.get_metric,
                      valid_sets=val,
                      num_boost_round=5000,
                      callbacks = [early_stopping,
                                   lgb.record_evaluation(evals_result)])
        else:
            model = lgb.LGBMClassifier(                                                     #create model with WCE with best params
                        num_leaves = int(optimum["num_leaves"]),
                        learning_rate = optimum["learning_rate"],
                        max_depth = int(optimum["max_depth"]),
                        max_bin = int(optimum["max_bin"]),
                        min_child_samples = int(optimum["min_data_in_leaf"]),
                        reg_alpha = optimum["lambda_l1"],
                        reg_lambda = optimum["lambda_l2"],
                        feature_fraction = optimum["feature_fraction"],
                        class_weight = "balanced",
                        seed = np.random.randint(0, 100),
                        n_estimators = 5000
                        )
            
            model.fit(x_train, y_train, 
                     eval_metric="cross_entropy",
                     eval_set=(x_val, y_val), 
                     callbacks = [early_stopping,
                                  lgb.record_evaluation(evals_result)])
            
        end = time.time()
        t = end - start                                                                     #get training time
        times.append(t)
        evals_result = evals_result["valid_0"]
        evals_result = evals_result.items()
        evals_result = list(evals_result)[0][1]
        rounds.append(evals_result)                                                         #save number of boosting iters
        
    return times, rounds

def eval_loss_time(x, y, loss_name, n_optima = 3, n_iters = 5):
    search = create_param_space(loss_name)                                                  #create param space
    times_box = []
    rounds_box = []
    
    for i in range(n_optima):                                                               #run eval on N different optima for robustness
        optimum = optimize_CV(x, y, loss_name, search)                                      #optimize
        objective = create_objective(loss_name, optimum, y)                                 #create loss wrapper
        times, rounds = measure_time(x, y, optimum, objective, iters = n_iters)             #measure times and boosting iters
        times_box += times
        rounds_box += rounds
        
    return times_box, rounds_box

###############################################################################

#define params for analysis
x, y  = molnet_loader("HIV", 0)
to_csv = True
n_optima = 3
n_iters = 5

###############################################################################

print("--Focal loss analysis")
times_fc, rounds_fc = eval_loss_time(x, y, "Focal_loss", n_optima, n_iters)

print("--LA loss analysis")
times_la, rounds_la = eval_loss_time(x, y, "LA_loss", n_optima, n_iters)

print("--EQ loss analysis")
times_eq, rounds_eq = eval_loss_time(x, y, "EQ_loss", n_optima, n_iters)

print("--LDAM loss analysis")
times_ldam, rounds_ldam = eval_loss_time(x, y, "LDAM_loss", n_optima, n_iters)

print("--WCE loss analysis")
times_wce, rounds_wce = eval_loss_time(x, y, None, n_optima, n_iters)

output = pd.DataFrame({
                            "WCE_times": times_wce,
                            "WCE_rounds": rounds_wce,
                            
                            "FC_times": times_fc,
                            "FC_rounds": rounds_fc,
                            
                            "LA_times": times_la,
                            "LA_rounds": rounds_la,
                            
                            "EQ_times": times_eq,
                            "EQ_rounds": rounds_eq,
                            
                            "LDAM_times": times_ldam,
                            "LDAM_rounds": rounds_ldam})

del (times_fc, rounds_fc, times_la, rounds_la, times_eq, rounds_eq, times_ldam,
    rounds_ldam, times_wce, rounds_wce)

if to_csv is True:
    output.to_csv("output/convergence_summary.csv")


