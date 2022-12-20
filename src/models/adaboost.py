from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from src import config
from src.models.abstractmodel import GaspipelineModelTrainer
from src.preprocess.featureextraction import GasPipelineFeatureExtraction


class AdaBoostTrainer(GaspipelineModelTrainer):
    best_parameters = {
        'balance_dataset': True,
        'feature_reduction': True,
        'scale_features': True,
        'feature_n_components': 12
    }

    tuning_parameters = {

    }

    def __init__(self):
        super().__init__()
        self.model = GasPipelineAdaBoost(**self.best_parameters)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def tune(self):
        tuned_model = GridSearchCV(self.model, self.tuning_parameters, cv=5, verbose=config.verbosity, n_jobs=-1)
        tuned_model = tuned_model.fit(self.x_train, self.y_train)

        return tuned_model.cv_results_

    def get_model(self):
        return self.model


class GasPipelineAdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, balance_dataset=False, feature_reduction=False, scale_features=False, feature_n_components=1, **kwargs):
        self.balance_dataset = balance_dataset
        self.feature_reduction = feature_reduction
        self.scale_features = scale_features
        self.feature_n_components = feature_n_components
        self.feature_extraction = GasPipelineFeatureExtraction(self.feature_reduction, self.scale_features, self.feature_n_components)
        self.ada = AdaBoostClassifier(**kwargs)

    def fit(self, X, y):
        if self.balance_dataset:
            X, y = SMOTE().fit_resample(X, y)
        X = self.feature_extraction.fit_transform(X, y)
        self.ada.fit(X, y)

    def predict(self, X):
        X = self.feature_extraction.transform(X)
        return self.ada.predict(X)

    def score(self, X, y, sample_weight=None):
        X = self.feature_extraction.transform(X)
        return self.ada.score(X, y, sample_weight)

    def set_params(self, **params):
        self.__init__(**params)
        return self