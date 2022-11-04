import pandas as pd
from scipy.stats import ttest_ind_from_stats
import numpy as np

def t_test(arr1, arr2, n1, n2):
    
    mean1 = np.mean(arr1)
    std1 = np.std(arr1)
    
    mean2 = np.mean(arr2)
    std2 = np.std(arr2)
    
    a, b = ttest_ind_from_stats(mean1,
                                std1,
                                n1,
                                mean2,
                                std2,
                                n2,
                                equal_var=False,
                                alternative="less")
    return b


#datasets = ["HIV", "Tox21", "MUV", "phos", "ntp"]
datasets = ["hts"]
losses = ["fc", "la", "ldam", "eq"]
p_val_box = np.empty((5, 8, 4))
col_names = ["ROC-AUC",
	    "PR-AUC",
	    "ACCURACY",
	    "BALANCED ACCURACY",
	    "PRECISION",
	    "RECALL",
	    "F1 SCORE",
	    "MCC"]

for i in range(len(datasets)):
    wce = pd.read_csv("../output/" + datasets[i] + "_wce.csv")
    wce = np.array(wce)[:,1:]
    for j in range(len(losses)):
        loss = pd.read_csv("../output/" + datasets[i] + "_" + losses[j] + ".csv")
        loss = np.array(loss)[:,1:]
        for k in range(8):
            a = wce[:,k]
            b = loss[:,k]
            if datasets[i] == "phos" or datasets[i] == "ntp":
                p_val_box[i, k, j] = t_test(a, b, 5, 5)
            else:
                p_val_box[i, k, j] = t_test(a, b, 40, 40)

fc = pd.DataFrame(p_val_box[:,:,0], columns = col_names)
fc["Dataset"] = datasets
fc.to_csv("../output/fc_p_analysis.csv")

la = pd.DataFrame(p_val_box[:,:,1], columns = col_names)
la["Dataset"] = datasets
la.to_csv("../output/la_p_analysis.csv")

ldam = pd.DataFrame(p_val_box[:,:,2], columns = col_names)
ldam["Dataset"] = datasets
ldam.to_csv("../output/ldam_p_analysis.csv")

eq = pd.DataFrame(p_val_box[:,:,3], columns = col_names)
eq["Dataset"] = datasets
eq.to_csv("../output/eq_p_analysis.csv")

