import numpy as np
from scipy import optimize as opt

class loss_function:
    
    def __init__(self, y_true, class_weight = "balanced"):
        
        self.y_true = y_true
        
        if class_weight == "balanced":
            majority = len(y_true) - np.sum(y_true)
            minority = np.sum(y_true)
        
            majority = (1 / majority) * (len(y_true)/2)
            minority = (1 / minority) * (len(y_true)/2)
            self.alpha_m = minority / (majority + minority)
            self.alpha_M = majority / (majority + minority)
        else:
            self.alpha_m = 1
            self.alpha_M = 1
            
        self.init_score = 0    
                
    def get_init_score(self):
        res = opt.minimize_scalar(
            lambda p: -self.__call__(self.y_true, p).sum(),
            bounds=(-10, 10),
            method='bounded'
        )
        self.init_score = res.x        

###############################################################################

class LDAM_loss(loss_function):
    "based on https://github.com/kaidic/LDAM-DRW"
    def __init__(self, y_true, max_m = 0.5, class_weight = "balanced"):
        
        super(LDAM_loss, self).__init__(y_true, class_weight)
        
        majority = len(y_true) - np.sum(y_true)
        minority = np.sum(y_true)
        cls_num_list = [majority, minority]
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = m_list
        
    def __call__(self, y_true, y_pred):
        
        batch_m = y_true.copy()
        batch_m[np.where(batch_m==0)] = self.m_list[0]
        batch_m[np.where(batch_m==1)] = self.m_list[1]
        
        y_m = y_pred - batch_m

        # if condition is true, return x_m[index], otherwise return x[index]
        index_bool = y_true > 0
        output = np.where(index_bool, y_m, y_pred)
        output = output + self.init_score

        p = 1/(1+np.exp(-output))
        q = 1-p
        pos_loss = np.log(p) * self.alpha_m
        neg_loss = np.log(q) * self.alpha_M
        return y_true * pos_loss + (1 - y_true) * neg_loss

###############################################################################

class Focal_loss(loss_function):
    "based on https://github.com/artemmavrin/focal-loss/tree/7a1810a968051b6acfedf2052123eb76ba3128c4"
    def __init__(self, y_true, gamma = 2.0, class_weight="balanced"):
        
        super(Focal_loss, self).__init__(y_true, class_weight)
        self.gamma = gamma
        
    def __call__(self, y_true, y_pred):
        
        y_pred = y_pred + self.init_score
        
        p = 1/(1+np.exp(-y_pred))
        q = 1-p
        
        pos_loss = (q ** self.gamma) * np.log(p) * self.alpha_m
        neg_loss = (p ** self.gamma) * np.log(q) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss

###############################################################################

class EQ_loss(loss_function):
    "based on https://github.com/tztztztztz/eqlv2"
    def __init__(self,
                 y_true,
                 gamma=0.3,
                 mu=0.8,
                 ):
        
        super(EQ_loss, self).__init__(y_true, "other")
        self.gamma = gamma
        self.mu = mu
        self.pos_neg = None
        self.pos_grad = 0
        self.neg_grad = 0
        self.grad_box = []
        self.w_box = []
        self.w_box_2 = []
        
    def __call__(self,
                 y_true,
                 y_pred,
                 training = True):
        
        pos_w, neg_w = self.get_weight()

        weight = pos_w * y_true + neg_w * (1 - y_true)
        
        y_pred = y_pred + self.init_score
        
        p = 1/(1+np.exp(-y_pred))
        q = 1-p
        
        pos_loss = np.log(p)
        neg_loss = np.log(q)        
        
        cls_loss = y_true * pos_loss + (1 - y_true) * neg_loss
        
        cls_loss = cls_loss * weight
        
        if training is True:
            self.collect_grad(p, y_true)
        
        self.w_box.append(pos_w)
        self.w_box_2.append(neg_w)
        
        return cls_loss

    def collect_grad(self, p, y_true):
        grad = p * (1 - p)
        grad = np.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = np.sum(grad * y_true)
        neg_grad = np.sum(grad * (1 - y_true))

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)
        self.grad_box.append(self.pos_neg)
        
    def get_weight(self):
        if self.pos_neg != None:
            neg_w = self.mapping_function(self.pos_neg)
            pos_w = 1 - neg_w
            #pos_w = 1 + self.alpha * (1 - neg_w)
        else:
            pos_w = 0.5
            neg_w = 0.5
        return pos_w, neg_w

    def mapping_function(self, x):
        return 1 / (1 + np.exp(-self.gamma * (x - self.mu)))

###############################################################################

class LA_loss(loss_function):
    "based on https://github.com/google-research/google-research/tree/master/logit_adjustment"
    def __init__(self, y_true, tau=1.0, class_weight = "balanced"):
        
        super(LA_loss, self).__init__(y_true, class_weight)
        
        self.tau = tau
        majority = len(y_true) - np.sum(y_true)
        minority = np.sum(y_true)
        self.pi_pos = minority / len(y_true)
        self.pi_neg = majority / len(y_true)
        
    
    def __call__(self, y_true, y_pred):
        
        y_pred = y_pred + self.init_score
        
        scale = self.pi_pos * y_true + self.pi_neg * (1 - y_true)
        scale = scale**self.tau + 1e-12
        y_pred = y_pred + np.log(scale)
        
        p = 1/(1+np.exp(-y_pred))
        q = 1-p
        
        pos_loss = np.log(p) * self.alpha_m
        neg_loss = np.log(q) * self.alpha_M
        
        return y_true * pos_loss + (1 - y_true) * neg_loss
        
   
        
        
        