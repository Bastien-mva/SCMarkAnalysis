import scanpy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyPLNmodels import PLN, PLNPCA



def get_Y_and_GT(number_of_classes = 28, number_of_dimension = 15000, number_of_samples = 20000):
    """
    Takes the first "number_of_classes" classes and the
    "number of dimension" variables which have the more variance.
    Y corresponds to the counts, and Gt to the ground truth.
    """
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()[:number_of_samples]
    GT = data.obs['standard_true_celltype_v5'][:number_of_samples]
    le = LabelEncoder()
    GT = le.fit_transform(GT)
    filter = GT< number_of_classes
    GT = GT[filter]
    Y = Y[filter]
    not_only_zeros = np.sum(Y, axis = 0)>0
    Y = Y[:,not_only_zeros]
    var = np.var(Y, axis = 0)
    most_variables = np.argsort(var)[-number_of_dimension:]
    Y = Y[:,most_variables]
    return Y,GT

Y,GT = get_Y_and_GT(number_of_classes = 5, number_of_dimension = 1000)
plnpca = PLNPCA(q = 5)
plnpca.fit(Y, verbose = True)
print(plnpca)
