from os import cpu_count

from numpy import concatenate, linspace, logspace
from scipy.stats import loguniform, norm, truncnorm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import balance_dataset, convert_binary_labels, remove_missing_values, scale_features
from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class SvmTrainer(GaspipelineModelTrainer):
    best_parameters = {
        'balance_dataset': False,
        'feature_reduction': False,
        'scale_features': True,
        'kernel': 'rbf',
        'cache_size': 4000,
        'shrinking': False
    }

    tuning_parameters = {
        'balance_dataset': [False],
        'feature_reduction': [False],
        'scale_features': [True],
        'kernel': ['rbf'],
        'cache_size': [1000, 2000, 4000, 8000],
        'shrinking': [True, False]
    }

    def __init__(self):
        super().__init__()
        self.model = GasPipelineSvc(verbose=config.verbosity, **self.best_parameters)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, cv=5, verbose=config.verbosity, n_jobs=-1)
        tuned_model = tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model

    def _preprocess_features(self, x_train, x_test, y_train, y_test):
        x_train, y_train = remove_missing_values(x_train, y_train)
        x_test, y_test = remove_missing_values(x_test, y_test)

        y_train = convert_binary_labels(y_train)
        y_test = convert_binary_labels(y_test)
        return x_train, x_test, y_train, y_test


class GasPipelineSvc(BaseEstimator, ClassifierMixin):
    def __init__(self, balance_dataset=False, feature_reduction=False, scale_features=False, **kwargs):
        self._scaler = None
        self._ica = None
        self._cca = None
        self._pca = None
        self.scale_features = scale_features
        self.feature_reduction = feature_reduction
        self.balance_dataset = balance_dataset
        self.svc = SVC(**kwargs)

    def fit(self, X, y):
        X, y = self.preprocess_train(X, y)
        self.svc.fit(X, y)

    def predict(self,X):
        X, y = self.preprocess_test(X)
        return self.svc.predict(X)

    def score(self, X, y, sample_weight=None):
        X, y = self.preprocess_test(X, y)
        return self.svc.score(X, y, sample_weight)

    def set_params(self, **params):
        self.scale_features = params.pop('scale_features', False)
        self.feature_reduction = params.pop('feature_reduction', False)
        self.balance_dataset = params.pop('balance_dataset', False)
        self.svc.set_params(**params)
        return self

    def preprocess_train(self, X, y):
        if self.balance_dataset:
            X, y = self.balance(X, y)

        if self.feature_reduction:
            X = self.reduction(X, y)

        if self.scale_features:
            X = self.scale(X)

        return X, y

    def preprocess_test(self, X, y=None):
        if self.feature_reduction:
            x_test_pca = self._pca.transform(X)
            x_test_cca = self._cca.transform(X)
            x_test_ica = self._ica.transform(X)
            X = concatenate((x_test_pca, x_test_cca, x_test_ica), axis=1)

        if self.scale_features:
            X = self._scaler.transform(X)

        return X, y

    def balance(self, X, y):
        return balance_dataset(X, y)

    def reduction(self, X, y):
        x_train_pca, pca = get_first_pca_feature(X)
        x_train_cca, cca = get_first_cca_feature(X, y)
        x_train_ica, ica = get_first_ica_feature(X)

        self._pca = pca
        self._cca = cca
        self._ica = ica

        X = concatenate((x_train_pca, x_train_cca, x_train_ica), axis=1)
        return X

    def scale(self, X):
        X, scaler = scale_features(X)
        self._scaler = scaler

        return X
