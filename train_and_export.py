import yaml
import importlib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle

from sklearn.base import TransformerMixin

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()

# Load your dataset
df = pd.read_csv("spamham.csv")

# Set your feature and target column names
TEXT_COLUMN = "Message"
LABEL_COLUMN = "Label"

# Encode labels (e.g., spam/ham -> 1/0)
le = LabelEncoder()
df[LABEL_COLUMN] = le.fit_transform(df[LABEL_COLUMN])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN], df[LABEL_COLUMN], test_size=0.2, random_state=42
)

# Load model.yaml
with open("config\model.yaml", "r") as f:
    config = yaml.safe_load(f)

grid_config = config["grid_search"]
model_configs = config["model_selection"]

best_model = None
best_score = 0
best_model_name = ""

# Loop through defined models
for module_key, model_def in model_configs.items():
    class_name = model_def["class"]
    module_name = model_def["module"]
    param_grid = model_def["search_param_grid"]

    # Dynamically load class
    model_class = getattr(importlib.import_module(module_name), class_name)

    # Check if classifier requires dense input
    requires_dense = class_name in ["GaussianNB"]

    # Create pipeline with optional DenseTransformer
    steps = [("tfidf", TfidfVectorizer())]
    if requires_dense:
        steps.append(("to_dense", DenseTransformer()))
    steps.append(("clf", model_class()))
    pipeline = Pipeline(steps)

    # Adjust param grid to match pipeline structure
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={f"clf__{k}": v for k, v in param_grid.items()},
        cv=grid_config["params"]["cv"],
        verbose=grid_config["params"]["verbose"],
        n_jobs=-1
    )

    print(f"\nTraining model: {class_name}")
    grid_search.fit(X_train, y_train)
    print(f"Best score for {class_name}: {grid_search.best_score_}")

    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_model_name = class_name

# Save best model
import os
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model: {best_model_name} with score: {best_score}")

# Final evaluation on unseen test data
y_pred = best_model.predict(X_test)
print("Saved to: artifacts/best_model.pkl")

print("\n========== FINAL TEST RESULTS ==========")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))