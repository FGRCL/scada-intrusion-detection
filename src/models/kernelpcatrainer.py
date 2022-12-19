from imblearn.over_sampling import SMOTE
from numpy import linspace, logspace, percentile, square, zeros
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.dataset import convert_binary_labels, remove_missing_values
from src.preprocess.transforms import BinaryLabelEncoder, GasPipelineFeatureExtraction


class KernelPcaTrainer(GaspipelineModelTrainer):
    best_parameters = {

    }

    tuning_parameters = {
        'anomaly_percentile': logspace(1e-5, 1e2, 10),
        'n_components': linspace(1, 100, 10, dtype=int),
        'kernel': ['linear', 'poly', 'rbf', 'cosine'],
        'balance_dataset': [False],
        'feature_reduction': [False],
        'scale_features': [True]
    }

    def __init__(self):
        super().__init__()
        self.model = KernelPcaClassifier(10, True, True, True, **self.best_parameters)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, cv=10, verbose=config.verbosity, n_jobs=-1)
        tuned_model = tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model

    def _preprocess_features(self, x_train, x_test, y_train, y_test):
        simple_imputer = SimpleImputer()
        binary_label_encoder = BinaryLabelEncoder()
        x_train = simple_imputer.fit_transform(x_train, y_train)
        x_test = simple_imputer.fit_transform(x_test, y_test)
        y_train = binary_label_encoder.transform(y_train)
        y_test = binary_label_encoder.transform(y_test)

        return x_train[:1000], x_test[:200], y_train[:1000], y_test[:200]


class KernelPcaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, anomaly_percentile=10, balance_dataset=False, feature_reduction=False, scale_features=False, **kwargs):
        self.anomaly_percentile = anomaly_percentile
        self._threshold = None
        self.balance_dataset = balance_dataset
        self.feature_reduction = feature_reduction
        self.scale_features = scale_features
        if self.balance_dataset:
            self.smote = SMOTE()
        self.feature_engineering = GasPipelineFeatureExtraction(feature_reduction, scale_features)
        self.kernel_pca = KernelPCA(n_jobs=-1, fit_inverse_transform=True, **kwargs)

    def fit(self, X, y):
        if self.balance_dataset:
            X, y = self.smote.fit_resample(X, y)
        X = self.feature_engineering.fit_transform(X, y)
        x_latent = self.kernel_pca.fit_transform(X, y)
        x_reconstructed = self.kernel_pca.inverse_transform(x_latent)
        mse = square(x_reconstructed - X).mean(axis=-1)
        self._threshold = percentile(mse, self.anomaly_percentile)

        return self

    def predict(self, X):
        X = self.feature_engineering.transform(X)
        x_latent = self.kernel_pca.transform(X)
        x_reconstructed = self.kernel_pca.inverse_transform(x_latent)
        mse = square(x_reconstructed - X).mean(axis=-1)
        y_pred = mse > self._threshold

        return y_pred

    def score(self, X, y, sample_weight=None):
        X = self.feature_engineering.transform(X)
        y_pred = self.predict(X)
        return f1_score(y, y_pred)

    def set_params(self, **params):
        self.anomaly_percentile = params.pop('anomaly_percentile', 10)
        self.scale_features = params.pop('scale_features', False)
        self.feature_reduction = params.pop('feature_reduction', False)
        self.balance_dataset = params.pop('balance_dataset', False)
        if self.balance_dataset:
            self.smote = SMOTE()
        self.feature_engineering = GasPipelineFeatureExtraction(self.feature_reduction, self.scale_features)
        self.kernel_pca.set_params(**params)
        return self
