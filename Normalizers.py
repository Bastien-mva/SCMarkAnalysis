from utils import log_normalization
from pyPLNmodels import PLNPCA, PLN
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA

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
class pcalognorm80(lognorm):
    label_name="log normalization with 80 PCs"

    def __init__(self):
        self.name = "log_norm_pca80"
        self.projected = False
        self.fitted = False
    def fit(self,Y):
        super().fit(Y)
        pca = PCA(n_components=80)
        self.logY = pca.fit_transform(self.logY)
class pcalognorm10(lognorm):
    label_name="log normalization with 10 PCs"

    def __init__(self):
        self.name = "log_norm_pca10"
        self.projected = False
        self.fitted = False

    def fit(self,Y):
        super().fit(Y)
        pca = PCA(n_components=10)
        self.logY = pca.fit_transform(self.logY)


class plnpca(normalizer, ABC):

    def fit(self, Y):
        self.model.fit(Y, O_formula="sum",tol = 0.0001)

    def get_normalized_matrix(self):
        if self.project is True:
            return self.model.best_model().get_projected_latent_variables(nb_dim = self.model.best_model()._q)
        else:
            return self.model.best_model().latent_variables


class plnpca_vlr_projected(plnpca):
    label_name = f"plnpca q={RANKS[0]} projected with ortho C in dim {RANKS[0]}"

    def __init__(self):
        super().__init__("plnpca_vlr_proj_norm")
        self.model = PLNPCA(ranks=[RANKS[0]])
        self.project = True


class plnpca_vlr_notprojected(plnpca):
    label_name = f"plnpca q={RANKS[0]} not projected"

    def __init__(self):
        super().__init__("plnpca_vlr_notproj_norm")
        self.model = PLNPCA(ranks=[RANKS[0]])
        self.project = False


class plnpca_lr_projected(plnpca):
    label_name = f"plnpca q={RANKS[1]} projected with orthoC in dim{RANKS[1]}"

    def __init__(self):
        super().__init__("plnpca_lr_proj_norm")
        self.model = PLNPCA(ranks=[RANKS[1]])
        self.project = True

class plnpca_lr_pcaprojected(plnpca):
    label_name = f"plnpca with q={RANKS[1]} projected with pca in dim{RANKS[1]}"

    def __init__(self):
        super().__init__("plnpca_lr_pcaproj_norm")
        self.model = PLNPCA(ranks=[RANKS[1]])
        self.project = True

    def get_normalized_matrix(self):
        return self.model.best_model().get_pca_projected_latent_variables(nb_dim = self.model.best_model()._q)



class plnpca_lr_notprojected(plnpca):
    label_name = f"plnpca with {RANKS[1]} not projected"

    def __init__(self):
        super().__init__("plnpca_lr_notproj_norm")
        self.model = PLNPCA(ranks=[RANKS[1]])
        self.project = False
