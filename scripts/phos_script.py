"""
Script to get the results for the Phosphatase dataset
"""
import sys
sys.path.append('..')
from misc.losses_eval import *
from misc.utils import *
from misc.loaders import moldata_loader

import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

###############################################################################

#define params for analysis
dataset = "phos"
task_n = 5
to_csv = True

###############################################################################

#run benchmarks on custom loss function
fc_box = eval_loss_moldata("Focal_loss", dataset, task_n)
fc_box = np.mean(fc_box, axis=1)

la_box = eval_loss_moldata("LA_loss", dataset, task_n)
la_box = np.mean(la_box, axis=1)

eq_box = eval_loss_moldata("EQ_loss", dataset, task_n)
eq_box = np.mean(eq_box, axis=1)

ldam_box = eval_loss_moldata("LDAM_loss", dataset, task_n)
ldam_box = np.mean(ldam_box, axis=1)

#run benchmarks on baseline
loss = None
wce_box = np.empty((5, task_n, 8))   
for j in range(task_n):
    train_x, train_y, test_x, test_y, val_x, val_y = moldata_loader(dataset, j)
    search = create_param_space(loss)
    optimum = optimize(train_x, train_y,
                           val_x, val_y,
                           loss, search)
    for i in range(5):
        model = train_sklearn_model(train_x, train_y, val_x, val_y, optimum)
        val_p = model.predict_proba(val_x)[:,1]
        test_p = model.predict_proba(test_x)[:,1]
            
        roc, pr, acc, bacc, pre, rec, f1, mcc = get_metrics(val_y, val_p, test_y, test_p)
        wce_box[i, j, 0] = roc
        wce_box[i, j, 1] = pr
        wce_box[i, j, 2] = acc
        wce_box[i, j, 3] = bacc
        wce_box[i, j, 4] = pre
        wce_box[i, j, 5] = rec
        wce_box[i, j, 6] = f1
        wce_box[i, j, 7] = mcc 
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


