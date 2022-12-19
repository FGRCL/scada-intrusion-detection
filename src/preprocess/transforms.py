from imblearn.over_sampling import SMOTE
from numpy import nonzero
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, StandardScaler
from typing_extensions import override
from webencodings import labels


class GasPipelineFeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, feature_reduction=False, scale_features=False):
        transforms = [
            ('identity', IdentityTransformer())
        ]

        if feature_reduction:
            transforms.append(
                ('feature_reduction', FeatureUnion([
                    ('pca', PCA(n_components=1)),
                    ('cca', CCAWrapper(n_components=1)),
                    ('ica', FastICA(n_components=1))
                ]))
            )

        if scale_features:
            transforms.append(
                ('scaler', StandardScaler())
            )

        self.pipeline = Pipeline(steps=transforms)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    @override
    def transform(self, X):
        return self.pipeline.transform(X)


class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        return self

    @override
    def transform(self, y):
        y = self._binarize(y)
        return y

    @staticmethod
    def _binarize(labels):
        malignant_labels = nonzero(labels)[0]
        labels[malignant_labels] = 1
        return labels


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