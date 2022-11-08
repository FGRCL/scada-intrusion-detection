from imblearn.over_sampling import SMOTE


def balance_dataset(features, labels):
    features_balanced, labels_balanced = SMOTE().fit_resample(features, labels)
    return features_balanced, labels_balanced
