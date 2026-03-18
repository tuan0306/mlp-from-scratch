import numpy as np

# mean quared error
def mse(y_true,y_pre):
    return np.mean((y_true-y_pre)**2)

def mse_prime(y_true,y_pre):
    N=y_true.shape[0]
    return 2*(y_pre-y_true)/N

# binary cross-entropy
def bce(y_true,y_pre):
    epsilon=1e-15
    y_pre=np.clip(y_pre,epsilon,1-epsilon)
    return -np.mean(y_true*np.log(y_pre)+(1-y_true)*np.log(1-y_pre))

def bce_prime(y_true,y_pre):
    epsilon=1e-15
    y_pre=np.clip(y_pre,epsilon,1-epsilon)
    N=y_true.shape[0]
    return (y_pre-y_true)/(y_pre*(1-y_pre)*N)
