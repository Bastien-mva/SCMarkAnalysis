import numpy as np
from pyPLNmodels import PLN, PLNPCA
from utils import get_sc_mark_data, log_normalization, remove_sequencing_depth
import matplotlib.pyplot as plt
from umap import UMAP
Y, GT = get_sc_mark_data(max_class=5, max_dim=1200, max_n = 4900)
plnpca = PLNPCA(ranks=[2, 6, 9, 12, 15, 18, 21, 24])
# plnpca.fit(Y, verbose=True, tol = 0.0001)
# print(plnpca)
# plnpca.show()
# plnpca.best_model().save_model("best_model")
best_rank = 24
plnpca = PLNPCA(ranks= best_rank)

plnpca.load_model_from_file(best_rank,"best_model")
plnpca[best_rank].fitted = True
latent_variables = plnpca[best_rank].get_projected_latent_variables(nb_dim = 24)
dr = UMAP()
dr_lognorm = dr.fit_transform(log_normalization(Y))
drlv = dr.fit_transform(latent_variables)


fig, axes = plt.subplots(3)
plnpca[best_rank].viz(color = GT, ax= axes[0])
axes[0].set_title("PCA with best_rank components")
axes[1].scatter(drlv[:,0], drlv[:,1], c= GT)
axes[1].set_title("UMAP on the 24 dimensions after PLNPCA")
axes[2].scatter(dr_lognorm[:,0], dr_lognorm[:,1], c = GT)
axes[2].set_title("UMAP on the log normalized data.")
plt.show()


