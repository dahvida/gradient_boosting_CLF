"""
Script to get the results for the HIV dataset
"""
import sys
sys.path.append('..')
from misc.losses_eval import eval_loss_molnet
from misc.utils import *

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
fc_box = fc_box[:,0]

la_box = eval_loss_molnet("LA_loss", dataset, task_n, roc_auc_score)
la_box = la_box[:,0]

eq_box = eval_loss_molnet("EQ_loss", dataset, task_n, roc_auc_score)
eq_box = eq_box[:,0]

ldam_box = eval_loss_molnet("LDAM_loss", dataset, task_n, roc_auc_score)
ldam_box = ldam_box[:,0]

#run benchmarks on baseline
x, y = molnet_loader(dataset, 0)
loss = None
search = create_param_space(loss)
optimum = optimize_CV(x, y, loss, search)
wce_box = []
for i in range(50):
        train_x, val_x, train_y, val_y = train_test_split(x, y,
                                                          stratify=y,
                                                          test_size=0.2)
        test_x, val_x, test_y, val_y = train_test_split(val_x, val_y,
                                                        stratify=val_y,
                                                        test_size=0.5)
        model = train_sklearn_model(train_x, train_y, val_x, val_y, optimum)
        preds = model.predict_proba(test_x)[:,1]
        wce_box.append(roc_auc_score(test_y, preds))

output = pd.DataFrame({
                            "WCE": wce_box,
                            "Focal loss": fc_box,
                            "Logit-adjusted loss": la_box,
                            "Equalization loss": eq_box,
                            "LDAM loss": ldam_box})

del (train_x, val_x, train_y, val_y, test_x, test_y, model, preds, fc_box, la_box,
     eq_box, ldam_box, loss, search, optimum, wce_box, x, y)

if to_csv is True:
    name = "output/" + dataset + "_summary.csv"
    output.to_csv(name)


