import numpy as np
from sklearn.model_selection import cross_val_score
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pyPLNmodels.VEM import PLN, PLNPCA
from utils import get_sc_mark_data
from umap import UMAP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

fitting_models = {
    "KNeighbors": KNeighborsClassifier(),
    "MLP": MLPClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "xgbc": XGBClassifier(),
    "logreg": LogisticRegression(),
    "SVC": SVC(),
}


fitting_scores = {
    model: {"proj": [], "notproj": []} for model in fitting_models.keys()
}
def plot_scores(fitting_scores, ranks):
    fig, ax = plt.subplots()
    for modelname, proj_and_not_proj in fitting_scores.items():
        ax.plot(ranks,proj_and_not_proj["proj"], label = modelname + "proj", linestyle = '--')
        ax.plot(ranks,proj_and_not_proj["notproj"], label = modelname + "notproj", linestyle = '-')
    plt.legend()

n = 3000
max_class = 15
cv = 8
dimension = 100
cv = 10
Y, GT, GT_names = get_sc_mark_data(max_n=n, max_class=max_class, max_dim=dimension)

rank_pca = [2, 7,10, 20,30, 60, 80]#, 80, 150]

pca = PLNPCA(ranks=rank_pca)

pca.fit(Y, tol=0.1)
for rank in rank_pca:
    pca[rank].save_model(str(rank))

# for rank in rank_pca:
#     pca[rank].load_model_from_file(str(rank))

fig, axes = plt.subplots(len(rank_pca), 2)

def get_score(fitting_model, X, y, cv):
    if isinstance(X, torch.Tensor):
        X = X.cpu()
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    score = np.mean(
        cross_val_score(fitting_model, X, y, cv=cv, scoring="balanced_accuracy")
    )
    return score

def get_plot_args(pcamodel, axe, cv):
    print('actual dim:', pcamodel._q)
    Y_proj = pcamodel.get_projected_latent_variables(pcamodel._q)
    Y_notproj = pcamodel.latent_variables
    dr = UMAP()
    dr_proj = dr.fit_transform(Y_proj)
    dr_not_proj = dr.fit_transform(Y_notproj)
    sns.scatterplot(x = dr_proj[:,0], y = dr_proj[:,1], hue=GT_names, ax=axe[0])
    axe[0].set_title(f"Projection with {pcamodel._q} axes after a projection")
    sns.scatterplot(x = dr_not_proj[:,0], y = dr_not_proj[:,1], hue=GT_names, ax=axe[1])
    axe[1].set_title(f"Projection with {pcamodel._q} axes")

    for name, model in fitting_models.items():
        score_proj = get_score(model, Y_proj, GT, cv)
        score_notproj = get_score(model, Y_notproj, GT, cv)
        fitting_scores[name]["proj"].append(score_proj)
        fitting_scores[name]["notproj"].append(score_notproj)


for (axe,pcamodel) in zip(axes,pca.models):
    get_plot_args(pcamodel, axe, cv)


plt.savefig("UMAP.pdf", format = "pdf")
plt.show()
plot_scores(fitting_scores, rank_pca)
plt.savefig("scores.pdf", format = "pdf")
plt.show()
