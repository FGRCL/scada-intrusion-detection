import numpy as np
from imblearn.over_sampling import SMOTE
from numpy import isnan, all
from sklearn.preprocessing import StandardScaler


def balance_dataset(features, labels):
    features_balanced, labels_balanced = SMOTE().fit_resample(features, labels)
    return features_balanced, labels_balanced


def remove_missing_values(features, labels):
    features_clean = features[all(features != None, axis=1)]
    labels_clean = labels[labels != None]
    return features_clean, labels_clean


def scale_features(features):
    scaler = StandardScaler()
    scaler.fit(features)
    features_scaled = scaler.transform(features)
    return features_scaled, scaler
