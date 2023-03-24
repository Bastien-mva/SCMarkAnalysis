import numpy as np
from pyPLNmodels import PLN, PLNPCA
from utils import get_sc_mark_data, log_normalization, remove_sequencing_depth
import matplotlib.pyplot as plt
from umap import UMAP
import pandas as pd
import seaborn as sns
max_dim = 120
Y, GT, GT_name = get_sc_mark_data(max_class=5, max_dim=max_dim, max_n = 490)
best_rank = 24

# plnpca = PLNPCA(ranks=[best_rank])
# plnpca.fit(Y, verbose=True, tol = 0.0001)
# plnpca.best_model().save_model("best_model")
# best_rank = 24
plnpca = PLNPCA(ranks= best_rank)

plnpca.load_model_from_file(best_rank,"best_model")
plnpca[best_rank].fitted = True
latent_variables = plnpca[best_rank].get_projected_latent_variables(nb_dim = 24)
dr = UMAP()
dr_lognorm = dr.fit_transform(log_normalization(Y))
drlv = dr.fit_transform(latent_variables)


fig, axes = plt.subplots(3, figsize =(25,15))

plnpca[best_rank].viz(color = GT_name, ax= axes[0])
axes[0].set_title(f"PCA on the {best_rank} components of the PLNPCA")
sns.scatterplot(x = drlv[:,0],y =  drlv[:,1], hue = GT_name, ax = axes[1])
axes[0].set_xlabel("PCA 1")
axes[0].set_ylabel("PCA 2")


axes[1].set_title(f"UMAP on the {best_rank} components of the PLNPCA")
sns.scatterplot(x = dr_lognorm[:,0], y = dr_lognorm[:,1], hue = GT_name, ax = axes[2])
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

axes[2].set_xlabel("UMAP 1")
axes[2].set_ylabel("UMAP 2")
axes[2].set_title("UMAP on the log normalized data")
plt.suptitle(f"Visualization of a subsample of scMARK data using the {max_dim} most variable genes.")
plt.savefig("viz.pdf", format = 'pdf')
plt.show()
