from os import cpu_count

from imblearn.over_sampling import SMOTE
from numpy import concatenate, linspace, logspace, percentile, square, zeros
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureextraction import GasPipelineFeatureExtraction
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class PcaTrainer(GaspipelineModelTrainer):
    best_parameters = {
        'anomaly_percentile': 1e-9,
        'n_components': 19,
        'whiten': False,
        'balance_dataset': False,
        'feature_reduction': False,
        'scale_features': True,
    }

    tuning_parameters = {
        'anomaly_percentile': logspace(-10, 2, 13),
        'n_components': [19],
        'whiten': [False],
        'balance_dataset': [False],
        'feature_reduction': [True, False],
        'scale_features': [True, False],
        'feature_n_components': linspace(1, 12, 5, dtype=int),
    }

    def __init__(self):
        super().__init__()
        self.model = PcaAnomalyDetection(**self.best_parameters)

    def train(self):
        self.model.fit(self.x_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, verbose=config.verbosity, cv=10, n_jobs=cpu_count() * 2)
        tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model


class PcaAnomalyDetection(BaseEstimator, ClassifierMixin):
    def __init__(self, anomaly_percentile=5, balance_dataset=False, feature_reduction=False, scale_features=False, feature_n_components=1, **kwargs):
        self._threshold = None
        self.anomaly_percentile = anomaly_percentile
        self.balance_dataset = balance_dataset
        self.feature_reduction = feature_reduction
        self.scale_features = scale_features
        self.feature_n_components = feature_n_components
        self.feature_extraction = GasPipelineFeatureExtraction(self.feature_reduction, self.scale_features, self.feature_n_components)
        self.pca = PCA(**kwargs)

    def fit(self, X, y=None):
        if self.balance_dataset:
            X, y = SMOTE().fit_resample(X, y)
        X = self.feature_extraction.fit_transform(X, y)
        x_pca = self.pca.fit_transform(X)
        x_reconstructed = self.pca.inverse_transform(x_pca)
        mse = square(x_reconstructed - X).mean(axis=-1)
        self._threshold = percentile(mse, self.anomaly_percentile)

    def predict(self, X):
        X = self.feature_extraction.transform(X)
        x_pca = self.pca.transform(X)
        x_reconstructed = self.pca.inverse_transform(x_pca)
        mse = square(x_reconstructed - X).mean(axis=-1)
        y_pred = zeros(mse.size)
        y_pred[mse > self._threshold] =1

        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)

    def set_params(self, **params):
        self.__init__(**params)
        return self
