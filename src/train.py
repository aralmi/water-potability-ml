import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42


def add_features(X):
    """
    Add engineered features used in the final model.
    """
    X = X.copy()

    X["Solids_log"] = np.log1p(X["Solids"])
    X["Hardness_log"] = np.log1p(X["Hardness"])

    X["ph_squared"] = X["ph"] ** 2
    X["chloramines_squared"] = X["Chloramines"] ** 2
    X["turbidity_squared"] = X["Turbidity"] ** 2

    X["ph_chloramines"] = X["ph"] * X["Chloramines"]
    X["ph_hardness"] = X["ph"] * X["Hardness"]
    X["organic_turbidity"] = X["Organic_carbon"] * X["Turbidity"]
    X["chloramines_organic"] = X["Chloramines"] * X["Organic_carbon"]

    X["solids_conductivity_ratio"] = X["Solids"] / (X["Conductivity"] + 1e-6)
    X["sulfate_hardness_ratio"] = X["Sulfate"] / (X["Hardness"] + 1e-6)
    X["conductivity_solids_ratio"] = X["Conductivity"] / (X["Solids"] + 1e-6)

    return X


def build_final_model():
    """
    Build final ML pipeline.

    Final model:
    - feature engineering
    - median imputation
    - SMOTE
    - RandomForestClassifier with tuned hyperparameters
    """
    model = ImbPipeline(steps=[
        ("features", FunctionTransformer(add_features, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(
            k_neighbors=3,
            random_state=RANDOM_STATE
        )),
        ("model", RandomForestClassifier(
            n_estimators=700,
            max_depth=6,
            min_samples_leaf=2,
            max_features=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return model


def train_model(model, X, y):
    """
    Train model on given data.
    """
    model.fit(X, y)
    return model