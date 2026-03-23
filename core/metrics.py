import numpy as np

def accuracy_score(y_true,y_pre):
    return np.sum(y_true==y_pre)/len(y_true)

def precision_score(y_true,y_pre):
    tp=np.sum((y_true==1) & (y_pre==1))
    fp=np.sum((y_true==0) & (y_pre==1))
    return tp/(tp+fp)

def recall_score(y_true,y_pre):
    tp=np.sum((y_true==1) & (y_pre==1))
    fn=np.sum((y_true==1) & (y_pre==0))
    return tp/(tp+fn)

def f1_score(y_true,y_pre):
    precision=precision_score(y_true,y_pre)
    recall=recall_score(y_true,y_pre)
    return 2*precision*recall/(precision+recall)