import math
import scanpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from utils import get_real_data, get_test_accuracy
import os
from tqdm import tqdm
from Normalizers import (
    pln,
    plnzero,
    lognorm,
    plnpca,
    plnpca_vlr_projected,
    plnpca_vlr_notprojected,
    plnpca_lr_notprojected,
    plnpca_lr_projected,
    RANKS,
)
from umap import UMAP

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
if len(RANKS)>2:
    raise Exception("Trop de rang vont etre pris en compte")

def test_dimension(max_dim, plot=False):
    Y, GT = get_real_data(max_n=n, max_class=8, max_dim=max_dim)
    ## log normalization
    lognorm_score = get_test_accuracy(log_normalization(Y), GT)

    #### pca
    pca = PLNPCA(ranks=RANKS)
    pca.fit(Y, O_formula="sum")
    latent_pca_first = pca[RANKS[0]].latent_variables
    pca_score_first = get_test_accuracy(latent_pca_first, GT)

    latent_pca_second = pca[RANKS[1]].latent_variables
    pca_score_second = get_test_accuracy(latent_pca_second, GT)

    latent_pca_proj = pca[RANKS[1]].projected_latent_variables
    pca_score_proj = get_test_accuracy(latent_pca_proj, GT)

    ### pln with sum formula for O
    pln = PLN()
    pln.fit(Y, O_formula="sum")
    latent = pln.latent_variables
    pln_score = get_test_accuracy(latent, GT)

    ### pln without sum formula
    plnzero = PLN()
    plnzero.fit(Y, O_formula=None)
    latent_zero = plnzero.latent_variables
    plnzero_score = get_test_accuracy(latent_zero, GT)

    if plot is True:
        dr = UMAP()

        drlogY = dr.fit_transform(Y)
        drlatent_pca_first = dr.fit_transform(latent_pca_first)
        drlatent_pca_second = dr.fit_transform(latent_pca_second)
        drlatent = dr.fit_transform(latent)
        drlatent_zero = dr.fit_transform(latent_zero)

        fig, axes = plt.subplots(5)
        axes[0].scatter(drlogY[:, 0], drlogY[:, 1], c=GT)
        axes[0].legend()
        axes[0].set_title("UMAP after log normalization")

        axes[1].scatter(drlatent[:, 0], drlatent[:, 1], c=GT)
        axes[1].legend()
        axes[1].set_title("UMAP after normalization with pln")

        axes[2].scatter(drlatent_pca_first[:, 0], drlatent_pca_first[:, 1], c=GT)
        axes[2].legend()
        axes[2].set_title(f"UMAP after normalization with plnpca with rank{RANKS[0]}")

        axes[3].scatter(drlatent_pca_second[:, 0], drlatent_pca_second[:, 1], c=GT)
        axes[3].legend()
        axes[3].set_title(f"UMAP after normalization with plnpca with rank{RANKS[1]}")

        axes[4].scatter(drlatent_zero[:, 0], drlatent_zero[:, 1], c=GT)
        axes[4].legend()
        axes[4].set_title("UMAP after normalization with pln zero")
        plt.show()
    return (
        lognorm_score,
        pca_score_first,
        pca_score_second,
        pca_score_proj,
        pln_score,
        plnzero_score,
    )


def append_scores(method, new_score):
    method["xgb"].append(new_score["xgb"])
    method["svm"].append(new_score["svm"])


def test_dimensions(max_dims, plot=False):

    scores_lognorm = {"xgb": [], "svm": [], "name": "lognorm", "linestyle": "-"}
    scores_pca_first = {
        "xgb": [],
        "svm": [],
        "name": f"plnpca{RANKS[0]}",
        "linestyle": "--",
    }
    scores_pca_second = {
        "xgb": [],
        "svm": [],
        "name": f"plnpca{RANKS[1]}",
        "linestyle": "dotted",
    }
    scores_pca_proj = {
        "xgb": [],
        "svm": [],
        "name": f"plnpca_projected_dim{RANKS[1]}",
        "linestyle": (5, (10, 3)),
    }
    scores_pln = {"xgb": [], "svm": [], "name": "pln", "linestyle": "dashdot"}
    scores_plnzero = {
        "xgb": [],
        "svm": [],
        "name": "plnzero",
        "linestyle": (0, (1, 10)),
    }

    for max_dim in max_dims:
        print("Dimension :", max_dim)
        (
            new_lognorm_score,
            new_pca_score_first,
            new_pca_score_second,
            new_pca_score_proj,
            new_pln_score,
            new_plnzero_score,
        ) = test_dimension(max_dim, plot)
        append_scores(scores_lognorm, new_lognorm_score)
        append_scores(scores_pca_first, new_pca_score_first)
        append_scores(scores_pca_second, new_pca_score_second)
        append_scores(scores_pca_proj, new_pca_score_proj)
        append_scores(scores_pln, new_pln_score)
        append_scores(scores_plnzero, new_plnzero_score)

    return [
        scores_lognorm,
        scores_pca_first,
        scores_pca_second,
        scores_pca_proj,
        scores_pln,
        scores_plnzero,
    ]


def plot_res(res, dims):
    fig, ax = plt.subplots(figsize=(15, 15))
    for score in res:
        label = score["name"]
        to_plot_xgb = list(score["xgb"])
        to_plot_svm = list(score["svm"])
        ax.plot(
            dims,
            to_plot_xgb,
            label=label + "xgb",
            color="blue",
            linestyle=score["linestyle"],
        )
        ax.plot(
            dims,
            to_plot_svm,
            label=label + "svm",
            color="red",
            linestyle=score["linestyle"],
        )
    ax.set_xlabel("Number of genes took")
    ax.set_ylabel("Balanced accuracy score")
    ax.legend()
    plt.savefig("accuracy_score.pdf", format="pdf")
    plt.show()


class launching_arguments:
    def __init__(self, n, max_class, cv, normalizers):
        self.n = n
        self.max_class = max_class
        self.cv = cv
        self.normalizers = normalizers

    def check_if_normalizer_already_launched_dim(self, dim, my_normalizer):
        return os.path.exists(self.get_dim_normalizer_path_of_file(dim, my_normalizer))

    @property
    def main_directory(self):
        str_ranks = f"r0_{RANKS[0]}_r1_{RANKS[1]}"
        return f"results_simu/n_{self.n}_maxclass_{self.max_class}_cv_{self.cv}_{str_ranks}"

    def dim_directory(self, dim):
        return f"{self.main_directory}/dim_{dim}"

    def get_dim_normalizer_path_of_file(self, dim, my_normalizer):
        return f"{self.dim_directory(dim)}/{my_normalizer.name}"

    def save_normalizer_score(self, dim, my_normalizer, score):
        path_of_file = self.dim_directory(dim)
        os.makedirs(path_of_file, exist_ok=True)
        with open(path_of_file + f"/{my_normalizer.name}", "wb") as fp:
            pickle.dump(score, fp)

    def get_back_normalizer_score(self, dim, my_normalizer):
        path_of_file = self.get_dim_normalizer_path_of_file(dim, my_normalizer)
        with open(path_of_file, "rb") as fp:
            score = pickle.load(fp)
        return score


def get_proccessed_raw_dict(raw_dict):
    print('raw dict ', raw_dict)
    process_dict = {key: {"xgb": [], "svm": []} for key in raw_dict.keys()}

    for normalizer_name, list_scores in raw_dict.items():
        for score in list_scores:
            process_dict[normalizer_name]["xgb"].append(score["xgb"])
            process_dict[normalizer_name]["svm"].append(score["svm"])
    return process_dict

class plot_args:
    def __init__(self, normalizers, la):
        self.normalizers = normalizers
        self.raw_dict_of_scores = {normalizer.name: [] for normalizer in self.normalizers}
        self.la = la

    @property
    def processed_dict_of_scores(self):
        return get_proccessed_raw_dict(self.raw_dict_of_scores)

    def plot(self, dimensions):

        fig, axes = plt.subplots(2,figsize = (15, 15))
        for my_normalizer,dict_of_scores in zip(self.normalizers,self.processed_dict_of_scores.values()):
            axes[0].plot(dimensions, dict_of_scores["xgb"], label = my_normalizer.label_name)
            axes[1].plot(dimensions, dict_of_scores["svm"], label = my_normalizer.label_name)
        axes[0].set_title("xgboost scores")
        axes[1].set_title("svm scores")
        plt.legend()
        plt.show()

def launch_dimension(la, pa, dimension):
    Y, GT = get_real_data(max_n=la.n, max_class=la.max_class, max_dim=dimension)
    for my_normalizer in la.normalizers:
        if (
            la.check_if_normalizer_already_launched_dim(dimension, my_normalizer)
            is True
        ):
            print(
                f"get back data for dim={dimension} and normalizer={my_normalizer.name}"
            )
            score = la.get_back_normalizer_score(dimension, my_normalizer)
            pa.raw_dict_of_scores[my_normalizer.name].append(score)
        else:
            normalized_matrix = my_normalizer.fit_and_get_normalized_matrix(Y)
            score = get_test_accuracy(normalized_matrix, GT, cv=la.cv)
            pa.raw_dict_of_scores[my_normalizer.name].append(score)
            la.save_normalizer_score(dimension, my_normalizer, score)


n = 3000
max_class = 15
cv = 8
dimensions = np.arange(80,200, 5)

if np.max(RANKS) > np.max(dimensions):
    raise Exception("ranks are higher than dimensions")

my_normalizers = [
    pln(),
    plnzero(),
    plnpca_lr_projected(),
    plnpca_vlr_projected(),
    plnpca_lr_notprojected(),
    plnpca_vlr_notprojected(),
    lognorm(),
]
la = launching_arguments(n, max_class, cv, my_normalizers)
pa = plot_args(my_normalizers, la)
for dimension in tqdm(dimensions):
    launch_dimension(la, pa, dimension)
pa.plot(dimensions)
