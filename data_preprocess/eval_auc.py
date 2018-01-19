# encoding=utf-8

import pickle
import numpy as np
import sklearn.metrics as metrics

with open('../cache/code_dict.pkl', 'rb') as file:
    code_dict = pickle.load(file)
    code_dict = {id: code for code, id in code_dict.items()}

with open('../cache/scores.pkl', 'rb') as file:
    y_scores, y_trues = pickle.load(file)
    for i in range(y_scores.shape[1]):
        y_score = y_scores[:, i]
        y_true = y_trues[:, i]
        try:
            auc = metrics.roc_auc_score(y_true, y_score)
        except Exception as e:
            print('label %s auc %.3f' % (code_dict[i].upper(), 0), 'no True samples in this code')
        else:
            print('label %s auc %.3f' % (code_dict[i].upper(), auc))
