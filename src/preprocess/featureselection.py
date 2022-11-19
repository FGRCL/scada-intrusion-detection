from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, PCA


def get_first_pca_feature(x, seed=2020):
    pca_95 = PCA(n_components=1, random_state=seed)
    pca_95.fit(x)
    x_pca_95 = pca_95.transform(x)

    return x_pca_95, pca_95


def get_first_cca_feature(x, y):
    cca = CCA(n_components=1)
    cca.fit(x, y)
    x_cca, y_cca = cca.transform(x, y)

    return x_cca, cca


def get_first_ica_feature(x, seed=12):
    ica = FastICA(n_components=1, random_state=seed)
    x_ica = ica.fit_transform(x)

    return x_ica, ica
