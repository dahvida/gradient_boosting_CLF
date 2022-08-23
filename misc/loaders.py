"""
Loader functions to prepare the datasets for modelling
"""
import sys
sys.path.append('..')
import numpy as np
from rdkit import Chem
import pandas as pd
from misc.utils import get_ECFP

###############################################################################

def molnet_loader(dataset_name, task_id):
    dataset_name = dataset_name.lower()
    db = pd.read_csv("../datasets/" + dataset_name + ".csv")
    task_name = list(db.columns)[task_id]
    db = db.loc[:, ["smiles", task_name]]    
    db.dropna(inplace=True)
    
    smiles = list(db["smiles"])
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    x = get_ECFP(mols)
    y = np.array(db[task_name])
    
    return x, y



def moldata_loader(dataset_name, task_id):
    dataset_name = dataset_name.lower()
    db = pd.read_csv("../datasets/" + dataset_name + ".csv")
    
    task_name = list(db.columns[2:-1])[task_id]
    db = db.loc[:, ["smiles", task_name, "split"]]
    db.dropna(inplace=True)
    
    train = db.loc[db["split"]=="train"]
    test = db.loc[db["split"]=="test"]
    val = db.loc[db["split"]=="validation"]
    
    X_tr = list(train["smiles"])
    X_tr = [Chem.MolFromSmiles(x) for x in X_tr]
    X_tr = get_ECFP(X_tr)
    y_tr = np.array(train[task_name])
        
    X_te = list(test["smiles"])
    X_te = [Chem.MolFromSmiles(x) for x in X_te]
    X_te = get_ECFP(X_te)
    y_te = np.array(test[task_name])
        
    X_v = list(val["smiles"])
    X_v = [Chem.MolFromSmiles(x) for x in X_v]
    X_v = get_ECFP(X_v)
    y_v = np.array(val[task_name])
    
    return X_tr, y_tr, X_te, y_te, X_v, y_v       
    
    
    
    


