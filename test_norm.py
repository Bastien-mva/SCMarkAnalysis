import seaborn as sns
from pyPLNmodels.VEM import PLN, PLNPCA
from utils import get_sc_mark_data
from umap import UMAP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

fitting_models = {"KNeighbors":KNeighborsClassifier(), "MLP": MLPClassifier(), "DecisionTree":DecisionTreeClassifier(), "xgbc":XGBClassifier(), "logreg":LogisticRegression(),"SVC":SVC()}
fitting_scores = {model:[] for model in fitting_models.keyrs()}

n = 3000
max_class = 15
cv = 8
dimension = 80

Y, GT , _ = get_sc_mark_data(max_n=n, max_class=max_class, max_dim=dimension)

rank_pca = [10,30,60,80, 150]


pca = PLNPCA(ranks =rank_pca)
pca.fit(Y, tol = 0.1)
for rank in rank_pca:
    pca.save_model(rank, str(rank))

for rank in rank_pca:
    pca[rank].load_model_from_file(str(rank))

fig, axes = plt.subplots(len(rank_pca),2)


def plot_project_and_get_score(pcamodel, axe):
    Y_proj = pcamodel.get_projected_latent_variables(pcamodel._q)
    Y_notproj = pcamodel.latent_variables
    dr = UMAP()
    dr_proj =  dr.fit_transform(Y_proj)
    dr_not_proj=  dr.fit_transform(Y_notproj)
    sns.scatterplot(dr_proj, hue = GT, ax = axe[0])
    axe[0].set_title(f"Projection with {pcamodel._q} axes")
    sns.scatterplot(dr_notproj, hue = GT, ax = axe[1])
    axe[1].set_title(f"Projection with {pcamodel._q} axes")

plot_project_and_get_score(pca.models, axes)





