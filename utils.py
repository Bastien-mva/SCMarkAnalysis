import numpy as np
from sklearn.preprocessing import LabelEncoder
import scanpy
import math
from xgboost import XGBClassifier
from sklearn import svm
import torch
from sklearn.model_selection import cross_val_score


def get_sc_mark_data(max_class=5, max_n=200, max_dim=100):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()[:max_n]
    GT_name = data.obs["standard_true_celltype_v5"][:max_n]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    unique, index = np.unique(GT, return_counts=True)
    enough_elem = index>50
    classes_with_enough_elem = unique[enough_elem]
    filter_bis = np.isin(GT,classes_with_enough_elem)
    mask = filter * filter_bis
    GT = GT[mask]
    GT_name = GT_name[mask]
    Y = Y[mask]
    GT = le.fit_transform(GT)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    var = np.var(Y, axis=0)
    most_variables = np.argsort(var)[-max_dim:]
    Y = Y[:, most_variables]
    return Y, GT, list(GT_name.values.__array__())


def log_normalization(Y):
    return np.log(Y + (Y == 0) * math.exp(-2))

def remove_sequencing_depth(Y):
    return Y/(np.log(np.sum(Y, axis = 1)).reshape(-1,1))


def get_test_accuracy(X, y, cv):
    xgb = XGBClassifier()
    svmclf = svm.SVC()
    if isinstance(X, torch.Tensor):
        X = X.cpu()
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    score_xgb = np.mean(cross_val_score(xgb, X, y, cv=cv, scoring="balanced_accuracy"))
    score_svm = np.mean(
        cross_val_score(svmclf, X, y, cv=cv, scoring="balanced_accuracy")
    )
    return {"xgb": score_xgb, "svm": score_svm}
