import sys
sys.path.append('..')
from losses.losses import *
from losses.loss_wrapper import loss_wrapper

import lightgbm as lgb
from sklearn.metrics import *
from scipy.special import expit
from hyperopt import tpe, hp, fmin, Trials
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

def get_ECFP(mols, radius = 2, nbits = 1024):
    array = np.empty((len(mols), nbits), dtype=np.float32)
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius, nbits) for x in mols]
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])
    return array

def train_model(train, val, objective, params):
    
    early_stopping = lgb.early_stopping(50, verbose=False)
    model = lgb.train(params ={
                            'num_leaves':           int(params["num_leaves"]),
                            'learning_rate':        params["learning_rate"],
                            'max_depth':            int(params["max_depth"]),
                            "max_bin":              int(params["max_bin"]),
                            'min_data_in_leaf':     int(params["min_data_in_leaf"]),
                            'lambda_l1':            params["lambda_l1"],
                            'lambda_l2':            params["lambda_l2"],
                            "feature_fraction":     params["feature_fraction"],
                            "verbosity":            -100,
                            "seed":                 np.random.randint(0, 100)
                        },
                      train_set = train,
                      fobj=objective.get_gradient,
                      feval=objective.get_metric,
                      valid_sets=val,
                      num_boost_round=1000,
                      callbacks = [early_stopping])

    return model

def train_sklearn_model(x_train, y_train, x_val, y_val, params):
        early_stopping = lgb.early_stopping(50, verbose=False)
        model = lgb.LGBMClassifier(
            num_leaves = int(params["num_leaves"]),
            learning_rate = params["learning_rate"],
            max_depth = int(params["max_depth"]),
            max_bin = int(params["max_bin"]),
            min_child_samples = int(params["min_data_in_leaf"]),
            reg_alpha = params["lambda_l1"],
            reg_lambda = params["lambda_l2"],
            feature_fraction = params["feature_fraction"],
            class_weight = "balanced",
            seed = np.random.randint(0, 100),
            n_estimators = 1000
            )
        model.fit(x_train, y_train, 
                 eval_metric="cross_entropy",
                 eval_set=(x_val, y_val), 
                 callbacks = [early_stopping])
        return model
        
    
def create_objective(loss_name, params, train_label):
    
    if loss_name == "LDAM_loss":
        loss_f = LDAM_loss(train_label, max_m=params["max_m"], class_weight=params["class_weight"])
        loss_f.get_init_score()
        objective = loss_wrapper(loss_f)
        
    elif loss_name == "LA_loss":
        loss_f = LA_loss(train_label, tau=params["tau"], class_weight=params["class_weight"])
        loss_f.get_init_score()
        objective = loss_wrapper(loss_f)
        
    elif loss_name == "EQ_loss":
        loss_f = EQ_loss(train_label, gamma=params["gamma"], mu=params["mu"])
        loss_f.get_init_score()
        objective = loss_wrapper(loss_f, clip=True, adaptive_weighting=True)
        
    elif loss_name == "Focal_loss":
        loss_f = Focal_loss(train_label, gamma=params["gamma"], class_weight=params["class_weight"])
        loss_f.get_init_score()
        objective = loss_wrapper(loss_f)    
        
    return objective


def create_param_space(loss_name):
    
        if loss_name == "LDAM_loss":
            loss_params = {
            'max_m':            hp.loguniform('max_m', -10, np.log(20)),
            'class_weight':     hp.choice('class_weight', ["balanced", "other"])
            }
        elif loss_name == "Focal_loss":
            loss_params = {
            'gamma':            hp.loguniform('gamma', -10, np.log(20)),
            'class_weight':     hp.choice('class_weight', ["balanced", "other"])
            }
        elif loss_name == "EQ_loss":
            loss_params = {
            'gamma':            hp.loguniform('gamma', -3, 2),
            'mu':               hp.loguniform('mu', -3, 2),
            }
        elif loss_name == "LA_loss":
            loss_params = {
            'tau':              hp.loguniform('tau', -10, np.log(20)),
            'class_weight':     hp.choice('class_weight', ["balanced", "other"])
            }
            
        params = {
            'num_leaves':           hp.qloguniform('num_leaves', np.log(10), np.log(10000), 1),
            'learning_rate':        hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'max_depth':            hp.quniform('max_depth', 3, 12, 1),
            "max_bin":              hp.quniform("max_bin", 100, 400, 5),
            'min_data_in_leaf':     hp.qloguniform('min_data_in_leaf', 0, 3, 1),
            'lambda_l1':            hp.loguniform('lambda_l1', -5, 2),
            'lambda_l2':            hp.loguniform('lambda_l2', -5, 2),
            "feature_fraction":     hp.uniform('feature_fraction', 0.1, 0.99),
        }
        
        if loss_name != None:
            for key in list(loss_params.keys()):
                params[key] = loss_params[key]
        
        return params


def optimize(
                 x_train, y_train,
                 x_val, y_val,
                 loss_name,
                 search,
                 iters = 20,
                 metric = roc_auc_score
                 ) -> dict:
    
            def model_eval(args):
            
            #define training loop
                params = args
                if loss_name != None:
                    objective = create_objective(loss_name, params, y_train)
                    train = lgb.Dataset(x_train, y_train)
                    val = lgb.Dataset(x_val, y_val)
                    model = train_model(train, val, objective, params)
                    preds = expit(model.predict(x_val)+objective.loss_function.init_score)
                else:
                    model = train_sklearn_model(x_train, y_train, x_val, y_val, params)
                    preds = model.predict_proba(x_val)[:,1]
                roc = metric(y_val, preds)
                return 1-roc

            trials = Trials()
            #get optimum hyperparameters
            optimum = fmin(
                fn = model_eval,
                space = search,
                algo = tpe.suggest,
                max_evals = iters,    
                trials = trials,
                )
    
            return optimum

def optimize_CV(
          x, y,
          loss_name,
          search,
          iters = 20,
          splits = 3,
          metric = roc_auc_score
        ):

        def model_eval(args):
            
            #define training loop
                params = args
                performance_box = []
                for i in range(splits):
                    x1, x2, y1, y2 = train_test_split(x, y, stratify=y, test_size=0.2)    
                    if loss_name != None:
                        objective = create_objective(loss_name, params, y1)
                        train = lgb.Dataset(x1, y1)
                        val = lgb.Dataset(x2, y2)
                        model = train_model(train, val, objective, params)
                        preds = expit(model.predict(x2)+objective.loss_function.init_score)
                    else:
                        model = train_sklearn_model(x1, y1, x2, y2, params)
                        preds = model.predict_proba(x2)[:,1]
                    performance_box.append(metric(y2, preds))
                
                return 1-np.mean(performance_box)

        trials = Trials()
        optimum = fmin(
                fn = model_eval,
                space = search,
                algo = tpe.suggest,
                max_evals = iters,    
                trials = trials,
                )
    
        return optimum















































