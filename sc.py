import scanpy
import numpy as np
from sklearn.preprocessing import LabelEncoder


data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
# print('array', data.X.toarray().shape)

X = data.X.toarray()
y = data.obs['standard_true_celltype_v5']
le = LabelEncoder()
y = le.fit_transform(y)
max_class = 5
filter = y< max_class
y = y[filter]
X = X[filter]
not_only_zeros = np.sum(X, axis = 0)>0
X = X[:,not_only_zeros]
var = np.var(X, axis = 0)
dim = 50
most_variables = np.argsort(var)[-dim:]
X = X[:,most_variables]
print('X :', X[50:400])
