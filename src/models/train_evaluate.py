
from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate_model(y_true, y_pred):
    """Évalue un modèle à partir des valeurs réelles et prédites."""
    score = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return score, conf_matrix
