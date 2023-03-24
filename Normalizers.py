from utils import log_normalization
from pyPLNmodels import PLNPCA, PLN
from abc import ABC, abstractmethod

RANKS = [10, 80]


class normalizer(ABC):
    def __init__(self, name):
        self.name = name
        self.projected = False
        self.fitted = False

    def fit_and_get_normalized_matrix(self, Y):
        self.fit(Y)
        self.fitted = True
        return self.get_normalized_matrix()

    @abstractmethod
    def get_normalized_matrix(self):
        pass


class pln(normalizer):
    label_name = "PLN with offsets"

    def __init__(self):
        super().__init__("pln_norm")
        self.model = PLN()

    def fit(self, Y):
        self.model.fit(Y, O_formula="sum",tol = 0.0001)

    def get_normalized_matrix(self):
        return self.model.latent_variables


class plnzero(normalizer):
    label_name = "PLN no offsets"

    def __init__(self):
        super().__init__("plnzero_norm")
        self.model = PLN()

    def fit(self, Y):  ## kwargs ????
        self.model.fit(Y, O_formula=None,tol = 0.0001)

    def get_normalized_matrix(self):
        return self.model.latent_variables


class lognorm(normalizer):
    label_name = "log normalization"

    def __init__(self):
        super().__init__("log_norm")

    def fit(self, Y):
        self.logY = log_normalization(Y)

    def get_normalized_matrix(self):
        return self.logY


class plnpca(normalizer, ABC):
    def fit(self, Y):
        self.model.fit(Y, O_formula="sum",tol = 0.0001)

    def get_normalized_matrix(self):
        if self.project is True:
            return self.model.best_model().projected_latent_variables
        else:
            return self.model.best_model().latent_variables


class plnpca_vlr_projected(plnpca):
    label_name = f"plnpca with rank {RANKS[0]} projected"

    def __init__(self):
        super().__init__("plnpca_vlr_proj_norm")
        self.model = PLNPCA(ranks=[RANKS[0]])
        self.project = True


class plnpca_vlr_notprojected(plnpca):
    label_name = f"plnpca with {RANKS[0]}"

    def __init__(self):
        super().__init__("plnpca_vlr_notproj_norm")
        self.model = PLNPCA(ranks=[RANKS[0]])
        self.project = False


class plnpca_lr_projected(plnpca):
    label_name = f"plnpca with {RANKS[1]} projected"

    def __init__(self):
        super().__init__("plnpca_lr_proj_norm")
        self.model = PLNPCA(ranks=[RANKS[1]])
        self.project = True


class plnpca_lr_notprojected(plnpca):
    label_name = f"plnpca with {RANKS[1]}"

    def __init__(self):
        super().__init__("plnpca_lr_notproj_norm")
        self.model = PLNPCA(ranks=[RANKS[1]])
        self.project = False
