from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

from src.config import f_score_beta


def get_classification_metrics(model, x_test, y_test):
    y_pred = model.predict(x_test)

    confusion = confusion_matrix(y_test, y_pred)
    f_score = fbeta_score(y_test, y_pred, f_score_beta)
    accuracy = accuracy_score(y_test, y_pred)

    return confusion, f_score, accuracy