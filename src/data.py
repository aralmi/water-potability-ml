from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str = "Potability") -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y