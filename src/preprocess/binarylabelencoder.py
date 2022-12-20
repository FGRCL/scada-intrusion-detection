from numpy import nonzero
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import override


class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y=None):
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