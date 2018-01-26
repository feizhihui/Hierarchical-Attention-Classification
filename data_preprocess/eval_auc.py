# encoding=utf-8

import pickle
import numpy as np
import sklearn.metrics as metrics

thres = 0.25

with open('../cache/code_dict.pkl', 'rb') as file:
    code_dict = pickle.load(file)
    code_dict = {id: code for code, id in code_dict.items()}

with open('../cache/scores.pkl', 'rb') as file, open('result.csv', 'w') as result:
    y_scores, y_trues = pickle.load(file)
    result.writelines('ICD9,AUC,Precision,Recall,F-score,P-Num,T-Num\n')
    print(len(y_scores))
    for i in range(y_scores.shape[1]):
        y_score = y_scores[:, i]
        y_true = y_trues[:, i]
        try:
            auc = metrics.roc_auc_score(y_true, y_score)
        except Exception as e:
            print('label %s auc %.3f' % (code_dict[i].upper(), 0), 'no True samples in this code')
        else:
            print('label %s auc %.3f' % (code_dict[i].upper(), auc))
            y_pred = (y_score > thres).astype(np.int32)
            P = metrics.precision_score(y_true, y_pred)
            R = metrics.recall_score(y_true, y_pred)
            F = metrics.f1_score(y_true, y_pred)
            pnum = np.sum(y_pred)
            tnum = np.sum(y_true)
            result.writelines(
                '{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n'.format(code_dict[i].upper(), auc, P, R, F, pnum, tnum))
