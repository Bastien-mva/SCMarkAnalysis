import numpy as np
from pyPLNmodels import PLN, PLNPCA
from utils import get_sc_mark_data


Y, GT = get_sc_mark_data(max_class=9, max_dim=100)
plnpca = PLNPCA(ranks=[2])
plnpca.fit(Y, verbose=True)
fig, ax = plt.plot()
