import numpy as np
from typing import Tuple
from scipy.misc import derivative

class loss_wrapper:
    
    def __init__(self,
                 loss_function,
                 use_minus_loss_as_objective = True,
                 clip = False,
                 adaptive_weighting = False
                     ):
        
        self.loss_function = loss_function
        self.sign = -1 if use_minus_loss_as_objective else 1
        self.clip = clip
        self.adaptive = adaptive_weighting

    def get_gradient(self,y_pred, y_true) -> Tuple[np.ndarray, np.ndarray]:
        y_true = y_true.get_label()
        
        temp = lambda x: self.loss_function(y_true, x)

        deriv1 = derivative(temp, y_pred, dx=1e-6, n=1) * self.sign 
        deriv2 = derivative(temp, y_pred, dx=1e-6, n=2) * self.sign 
        
        if self.clip is True:
            deriv1[np.where(deriv1 > 1)] = 1
            deriv1[np.where(deriv1 < -1)] = -1
            deriv2[np.where(deriv2 > 1)] = 1
            deriv2[np.where(deriv2 < -1)] = -1
        
        return deriv1, deriv2
    
    def get_metric(self, y_pred, y_true):
        y_true = y_true.get_label()
        if self.adaptive is False:
            loss = -np.mean(self.loss_function(y_true, y_pred))
        else:
            loss = -np.mean(self.loss_function(y_true, y_pred, training=False))
        
        return "loss", loss, False







