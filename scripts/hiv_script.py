"""
Script to get the results for the HIV dataset
"""
import sys
sys.path.append('..')
from misc.losses_eval import *
from misc.utils import *
from misc.loaders import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

###############################################################################

#define params for analysis
dataset = "HIV"
task_n = 1
to_csv = True

###############################################################################

#run benchmarks on custom loss function
fc_box = eval_loss_molnet("Focal_loss", dataset, task_n, roc_auc_score)
fc_box = np.mean(fc_box, axis=1)

la_box = eval_loss_molnet("LA_loss", dataset, task_n, roc_auc_score)
la_box = np.mean(la_box, axis=1)

eq_box = eval_loss_molnet("EQ_loss", dataset, task_n, roc_auc_score)
eq_box = np.mean(eq_box, axis=1)

ldam_box = eval_loss_molnet("LDAM_loss", dataset, task_n, roc_auc_score)
ldam_box = np.mean(ldam_box, axis=1)

#run benchmarks on baseline
x, y = molnet_loader(dataset, 0)
loss = None
search = create_param_space(loss)
optimum = optimize_CV(x, y, loss, search)
wce_box = np.empty((50, 1, 8))   
for i in range(50):
        train_x, val_x, train_y, val_y = train_test_split(x, y,
                                                          stratify=y,
                                                          test_size=0.2)
        test_x, val_x, test_y, val_y = train_test_split(val_x, val_y,
                                                        stratify=val_y,
                                                        test_size=0.5)
        model = train_sklearn_model(train_x, train_y, val_x, val_y, optimum)
        val_p = model.predict_proba(val_x)[:,1]
        test_p = model.predict_proba(test_x)[:,1]
            
        roc, pr, acc, bacc, pre, rec, f1, mcc = get_metrics(val_y, val_p, test_y, test_p)
        wce_box[i, 0, 0] = roc
        wce_box[i, 0, 1] = pr
        wce_box[i, 0, 2] = acc
        wce_box[i, 0, 3] = bacc
        wce_box[i, 0, 4] = pre
        wce_box[i, 0, 5] = rec
        wce_box[i, 0, 6] = f1
        wce_box[i, 0, 7] = mcc 
wce_box = np.mean(wce_box, axis=1)

#package
col_names = ["ROC-AUC",
	    "PR-AUC",
	    "ACCURACY",
	    "BALANCED ACCURACY",
	    "PRECISION",
	    "RECALL",
	    "F1 SCORE",
	    "MCC"]
fc_output = pd.DataFrame(fc_box, columns=col_names)
la_output = pd.DataFrame(la_box, columns=col_names)
eq_output = pd.DataFrame(eq_box, columns=col_names)
ldam_output = pd.DataFrame(ldam_box, columns=col_names)
wce_output = pd.DataFrame(wce_box, columns=col_names)

if to_csv is True:
    prefix = "../output/" + dataset + "_"
    fc_output.to_csv(prefix + "fc.csv")
    la_output.to_csv(prefix + "la.csv")
    eq_output.to_csv(prefix + "eq.csv")
    ldam_output.to_csv(prefix + "ldam.csv")
    wce_output.to_csv(prefix + "wce.csv")
    
    


