from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, PCA


def pca(x, seed=2020):
    pca_95 = PCA(n_components=1, random_state=seed)
    pca_95.fit(x)
    x_pca_95 = pca_95.transform(x)


    return x_pca_95


def cca(x, y):
    cca = CCA(n_components=1)
    cca.fit(x, y)
    x_cca, y_cca = cca.transform(x, y)
    return x_cca


def ica(x, seed=12):
    ICA = FastICA(n_components=1, random_state=seed)
    x_ica = ICA.fit_transform(x)

    return x_ica

def get_max_features(pca_features, cca_features, ica_features):
    pass