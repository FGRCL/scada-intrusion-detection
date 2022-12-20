from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, PCA
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from typing_extensions import override


class GasPipelineFeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, feature_reduction=False, scale_features=False, n_components=1):
        transforms = [
            ('identity', IdentityTransformer())
        ]

        if feature_reduction:
            transforms.append(
                ('feature_reduction', FeatureUnion([
                    ('pca', PCA(n_components=n_components)),
                    ('cca', CCAWrapper(n_components=1)),
                    ('ica', FastICA(n_components=n_components))
                ]))
            )

        if scale_features:
            transforms.append(
                ('scaler', StandardScaler())
            )

        self.pipeline = Pipeline(steps=transforms)

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    @override
    def transform(self, X):
        return self.pipeline.transform(X)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, input_array, y=None):
        return self

    @override
    def transform(self, X):
        return X


class CCAWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.cca = CCA(**kwargs)

    def fit(self, X, y):
        self.cca.fit(X, y)
        return self

    def transform(self, X):
        return self.cca.transform(X)