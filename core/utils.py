import numpy as np

def prob_to_class(y_pre_prob,thresold=0.5):
    return np.where(y_pre_prob>=thresold,1,0)