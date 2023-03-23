import math
import scanpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from utils import get_sc_mark_data, get_test_accuracy
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
if len(RANKS) > 2:
    raise Exception("Trop de rang vont etre pris en compte")


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
    print("raw dict ", raw_dict)
    process_dict = {key: {"xgb": [], "svm": []} for key in raw_dict.keys()}

    for normalizer_name, list_scores in raw_dict.items():
        for score in list_scores:
            process_dict[normalizer_name]["xgb"].append(score["xgb"])
            process_dict[normalizer_name]["svm"].append(score["svm"])
    return process_dict


class plot_args:
    def __init__(self, normalizers, la):
        self.normalizers = normalizers
        self.raw_dict_of_scores = {
            normalizer.name: [] for normalizer in self.normalizers
        }
        self.la = la

    @property
    def processed_dict_of_scores(self):
        return get_proccessed_raw_dict(self.raw_dict_of_scores)

    def plot(self, dimensions):

        fig, axes = plt.subplots(2, figsize=(15, 15))
        for my_normalizer, dict_of_scores in zip(
            self.normalizers, self.processed_dict_of_scores.values()
        ):
            axes[0].plot(
                dimensions, dict_of_scores["xgb"], label=my_normalizer.label_name
            )
            axes[1].plot(
                dimensions, dict_of_scores["svm"], label=my_normalizer.label_name
            )
        axes[0].set_title("xgboost scores")
        axes[1].set_title("svm scores")
        plt.legend()
        plt.show()


def launch_dimension(la, pa, dimension):
    Y, GT = get_sc_mark_data(max_n=la.n, max_class=la.max_class, max_dim=dimension)
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
dimensions = np.arange(80, 220, 5)
dimensions = np.concatenate((dimensions, np.arange(220,1500, 30)))
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
