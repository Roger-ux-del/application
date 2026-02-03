import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix





def build_pipeline(
    n_trees,
    numeric_features=None,
    categorical_features=None,
    max_depth=None,
    max_features="sqrt",
):
    """
    Construit et retourne un pipeline sklearn complet
    (preprocessing + RandomForest).
    """

    if numeric_features is None:
        numeric_features = ["Age", "Fare"]

    if categorical_features is None:
        categorical_features = ["Embarked", "Sex"]

    # Preprocessing numérique
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Preprocessing catégoriel
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Preprocessing global
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    # Pipeline final
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_trees,
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline
