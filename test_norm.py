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
from sklearn.neighbors import KNeighborsClassifier

fitting_models = {
    "KNeighbors": KNeighborsClassifier(),
    "MLP": MLPClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "xgbc": XGBClassifier(),
    # "logreg": LogisticRegression(),
    "SVC": SVC(),
}
colors = {"KNeighbors": "blue", "MLP": "black", "DecisionTree":"green", "xgbc":"red", "logreg":"pink", "SVC":"orange"}


fitting_scores = {
        model: {"proj": {"score":[], "var":[]} ,"notproj": {"score":[], "var":[]}, "pcaproj":{"score":[], "var":[]}} for model in fitting_models.keys()
}
pln_names = ["proj", "notproj", "pcaproj"]
pln_colors= {"proj":"blue", "notproj":"red", "pcaproj":"green"}


def plot_scores(fitting_scores, ranks):
    fig, axes = plt.subplots(len(fitting_models.keys()), figsize = (30,15))
    for ax, (modelname, proj_and_not_proj) in zip(axes,fitting_scores.items()):
        for pln_name in pln_names:
            IC = np.array(proj_and_not_proj[pln_name]["var"])*1.96/(np.sqrt(cv))
            mean = np.array(proj_and_not_proj[pln_name]["score"])
            ax.plot(ranks,mean , label = pln_name, linestyle = '-', color = pln_colors[pln_name])
            ax.plot(ranks, mean + IC , linestyle = '--', color = pln_colors[pln_name])
            ax.plot(ranks,mean -IC , linestyle = '--', color = pln_colors[pln_name])
        ax.legend()
        ax.set_title(modelname)


n = 10000
max_class = 28
dimension = 15000
cv =10
tol = 0.0001
Y, GT, GT_names = get_sc_mark_data(max_n=n, max_class=max_class, max_dim=dimension)
name= f"n{n}max_class{max_class}dimension{dimension}cv{cv}tol{tol}"
rank_pca = [2,3,4,5,6,7,8,9,10,20,30,60,80, 100, 125, 150, 200,250]

pca = PLNPCA(ranks=rank_pca)

pca.fit(Y, tol=tol)
for rank in rank_pca:
    pca[rank].save_model(str(rank)+ name)

for rank in rank_pca:
    pca[rank].load_model_from_file(str(rank)+ name)

fig, axes = plt.subplots(len(rank_pca), 3, figsize = (30,15))

def get_score(fitting_model, X, y, cv):
    if isinstance(X, torch.Tensor):
        X = X.cpu()
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    cvscore = cross_val_score(fitting_model, X, y, cv=cv, scoring="balanced_accuracy")
    score = np.mean(cvscore)
    variance = np.var(cvscore)
    return score, variance

def get_plot_args(pcamodel, axe, cv):
    print('actual dim:', pcamodel._q)
    Y_proj = pcamodel.get_projected_latent_variables(pcamodel._q)
    Y_notproj = pcamodel.latent_variables
    Y_pcaproj = pcamodel.get_pca_projected_latent_variables(pcamodel._q)
    # dr = UMAP()
    # dr_proj = dr.fit_transform(Y_proj)
    # dr_not_proj = dr.fit_transform(Y_notproj)
    # dr_pcaproj = dr.fit_transform(Y_pcaproj)

    # sns.scatterplot(x = dr_proj[:,0], y = dr_proj[:,1], hue=GT_names, ax=axe[0])
    # axe[0].set_title(f"Projection with {pcamodel._q} axes after a projection")
    # sns.scatterplot(x = dr_not_proj[:,0], y = dr_not_proj[:,1], hue=GT_names, ax=axe[1])
    # axe[1].set_title(f"Projection with {pcamodel._q} axes")
    # sns.scatterplot(x = dr_pcaproj[:,0], y = dr_pcaproj[:,1], hue=GT_names, ax=axe[2])
    # axe[1].set_title(f"Projection with {pcamodel._q} axes after pca projection")

    for name, model in fitting_models.items():
        score_proj, var_proj = get_score(model, Y_proj, GT, cv)
        score_notproj, var_notproj = get_score(model, Y_notproj, GT, cv)
        score_pcaproj, var_pcaproj = get_score(model, Y_pcaproj, GT, cv)
        fitting_scores[name]["proj"]["score"].append(score_proj)
        fitting_scores[name]["proj"]["var"].append(var_proj)
        fitting_scores[name]["notproj"]["score"].append(score_notproj)
        fitting_scores[name]["notproj"]["var"].append(var_notproj)
        fitting_scores[name]["pcaproj"]["score"].append(score_pcaproj)
        fitting_scores[name]["pcaproj"]["var"].append(var_pcaproj)


for (axe,pcamodel) in zip(axes,pca.models):
    get_plot_args(pcamodel, axe, cv)


# plt.savefig("UMAP.pdf", format = "pdf")
# plt.show()
plot_scores(fitting_scores, rank_pca)
plt.savefig("scores.pdf", format = "pdf")
plt.show()
