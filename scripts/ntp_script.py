"""
Script to get the results for the NTPase dataset
"""
import sys
sys.path.append('..')
from misc.losses_eval import eval_loss_moldata, get_metrics
from misc.utils import *
from misc.loaders import moldata_loader

import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

###############################################################################

#define params for analysis
dataset = "ntp"
task_n = 6
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
wce_box = np.empty((3, task_n, 5))    
for j in range(task_n):
    train_x, train_y, test_x, test_y, val_x, val_y = moldata_loader(dataset, j)
    search = create_param_space(loss)
    optimum = optimize(train_x, train_y,
                           val_x, val_y,
                           loss, search)
    for i in range(3):
        model = train_sklearn_model(train_x, train_y, val_x, val_y, optimum)
        val_p = model.predict_proba(val_x)[:,1]
        test_p = model.predict_proba(test_x)[:,1]
            
        acc, pre, rec, f1, roc = get_metrics(val_y, val_p, test_y, test_p)
        wce_box[i, j, 0] = acc
        wce_box[i, j, 1] = pre
        wce_box[i, j, 2] = rec
        wce_box[i, j, 3] = f1
        wce_box[i, j, 4] = roc     
wce_box = np.mean(wce_box, axis=1)

output = pd.DataFrame({
                            "WCE_accuracy": wce_box[:,0],
                            "WCE_precision": wce_box[:,1],
                            "WCE_recall": wce_box[:,2],
                            "WCE_f1": wce_box[:,3],
                            "WCE_ROC_AUC": wce_box[:,4],
                            
                            "FC_accuracy": fc_box[:,0],
                            "FC_precision": fc_box[:,1],
                            "FC_recall": fc_box[:,2],
                            "FC_f1": fc_box[:,3],
                            "FC_ROC_AUC": fc_box[:,4],
                            
                            "LA_accuracy": la_box[:,0],
                            "LA_precision": la_box[:,1],
                            "LA_recall": la_box[:,2],
                            "LA_f1": la_box[:,3],
                            "LA_ROC_AUC": la_box[:,4],
                            
                            "EQ_accuracy": eq_box[:,0],
                            "EQ_precision": eq_box[:,1],
                            "EQ_recall": eq_box[:,2],
                            "EQ_f1": eq_box[:,3],
                            "EQ_ROC_AUC": eq_box[:,4],
                            
                            "LDAM_accuracy": ldam_box[:,0],
                            "LDAM_precision": ldam_box[:,1],
                            "LDAM_recall": ldam_box[:,2],
                            "LDAM_f1": ldam_box[:,3],
                            "LDAM_ROC_AUC": ldam_box[:,4]})

del (train_x, val_x, train_y, val_y, test_x, test_y, model, val_p, test_p, acc, pre,
     rec, f1, roc, fc_box, la_box, eq_box, ldam_box, wce_box)

if to_csv is True:
    name = "output/" + dataset + "_summary.csv"
    output.to_csv(name)


