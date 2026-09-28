import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.entity.artifact_entity import ClassificationMetricArtifact


def calculate_metric(model, x, y) -> ClassificationMetricArtifact:
   """
   model: estimator
   x: input feature
   y: output feature
   """
   y = np.asarray(y).ravel()
   yhat = np.asarray(model.predict(x)).ravel()
   classification_metric = ClassificationMetricArtifact(
      accuracy_score=accuracy_score(y, yhat),
      f1_score=f1_score(y, yhat, average='binary', zero_division=0),
      recall_score=recall_score(y, yhat, average='binary', zero_division=0),
      precision_score=precision_score(y, yhat, average='binary', zero_division=0),
   )
   return classification_metric


def total_cost(y_true, y_pred):
   """
   This function takes y_ture, y_predicted, and prints Total cost due to misclassification
   """
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   cost = 10 * fp + 500 * fn
   return cost