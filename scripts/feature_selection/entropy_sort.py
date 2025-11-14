import numpy as np
import pandas as pd
from CICIDS2017.preprocessing import preprocess_cicids2017

# entropy_sort.py
# Requires a same-folder preprocessing.py that returns a pandas.DataFrame
# (function like load_data/load_dataset/read_data or a DataFrame attribute).
# Computes entropy and information gain (IG) for each feature and ranks features.


def entropy(series):
    """
    Compute the Shannon entropy of a pandas Series.

    Inputs:
        series (pd.Series): A categorical or discrete-valued series. NaN values are ignored.

    Returns:
        float: The Shannon entropy in bits (base-2). Returns 0.0 for empty series.
    """
    p = series.value_counts(normalize=True)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()

def _discretize_if_numeric(series, bins=10, qcut=True):
    """
    Discretize a numeric pandas Series into categorical bins, or return
    the input as a categorical dtype if it's non-numeric.

    Inputs:
        series (pd.Series): Input series to discretize.
        bins (int): Number of bins to create when discretizing numeric data.
        qcut (bool): If True use quantile-based bins (pandas.qcut). Otherwise use uniform-width bins (pandas.cut).

    Returns:
        pd.Series: A categorical series (either the binned numeric series or the original converted to 'category').
    """
    if pd.api.types.is_numeric_dtype(series):
        try:
            if qcut:
                return pd.qcut(series, q=bins, duplicates='drop')
            else:
                return pd.cut(series, bins=bins, duplicates='drop')
        except Exception:
            return pd.cut(series, bins=bins, duplicates='drop')
    return series.astype('category')

def conditional_entropy(feature, label, bins=10):
    """
    Compute the conditional entropy H(Y|X) where X is a feature and Y is the label.

    Inputs:
        feature (pd.Series): Feature series (may be numeric or categorical).
        label (pd.Series): Label/target series.
        bins (int): Number of bins to use when discretizing numeric feature values.

    Returns:
        float: The conditional entropy H(Y|X) in bits. Returns 0.0 when there are no valid rows.
    """
    # Note: preprocessing.py should have handled NaN removal.
    df = pd.concat([feature, label], axis=1)
    x = _discretize_if_numeric(df.iloc[:, 0], bins=bins)
    y = df.iloc[:, 1]
    # if label numeric, discretize to 2 bins (binary-ish)
    if pd.api.types.is_numeric_dtype(y):
        y = _discretize_if_numeric(y, bins=2, qcut=False)
    # Use number of non-null feature observations as total probability mass
    total = x.notna().sum()
    if total == 0:
        return 0.0
    ce = 0.0
    for lvl, idx in x.groupby(x, observed=False).groups.items():
        p_x = len(idx) / total
        suby = y.loc[idx]
        ce += p_x * entropy(suby)
    return ce

def information_gain(feature, label, bins=10):
    """
    Compute information gain IG(Y, X) = H(Y) - H(Y|X).

    Inputs:
        feature (pd.Series): Feature series used as X.
        label (pd.Series): Label/target series Y.
        bins (int): Binning parameter forwarded to conditional_entropy for numeric features.

    Returns:
        float: Information gain in bits.
    """
    return entropy(label.dropna()) - conditional_entropy(feature, label, bins=bins)

def _load_dataframe_from_preprocessing(logger=None):
    """
    Load a pandas DataFrame using the project's preprocessing module.

    Preference order:
    1. Call preprocessing.preprocess_cicids2017(logger=logger) if available.
    2. Call other common loader functions (trying to pass logger if accepted).
    3. Fall back to common DataFrame attributes on the module.

    Inputs:
        logger (optional): Logger instance passed to preprocessing functions that accept it.

    Returns:
        pd.DataFrame: Loaded and preprocessed dataset.

    Raises:
        RuntimeError: if no DataFrame-producing function or attribute is found on the preprocessing module.

    Raises RuntimeError if no DataFrame can be obtained.
    """
    # We import the preprocessing function directly (from CICIDS2017.preprocessing)
    # and call it. The caller may pass a logger; the function is expected to return
    # a pandas.DataFrame. Raise a clear error if the contract is violated.
    try:
        df = preprocess_cicids2017(logger=logger)
    except TypeError:
        # Some callers or older signatures may not accept logger; try without it
        df = preprocess_cicids2017()

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("preprocess_cicids2017 did not return a pandas.DataFrame")
    return df

def choose_label_column(df):
    """
    Choose the label/target column from a DataFrame using common candidate names.(Optional)

    Inputs:
        df (pd.DataFrame): Dataset to inspect.

    Returns:
        str: Column name selected as the label. Preference order is: 'label', 'Label', 'target', 'Target', 'class', 'Class', 'attack', 'Attack'.
             If none of these are present, returns the last column name in the DataFrame.
    """
    candidates = ['label', 'Label', 'target', 'Target', 'class', 'Class', 'attack', 'Attack']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column
    return df.columns[-1]

def rank_features(df, label_col=None, bins=10, save_csv=True, csv_path='feature_ranking.csv', logger=None):
    """
    Rank dataset features by information gain with respect to a binary label.

    Inputs:
        df (pd.DataFrame): Input dataset containing features and a label column.
        label_col (str|None): Name of the label column. If None, `choose_label_column` is used to pick one.
        bins (int): Number of bins for discretization when features are numeric.
        save_csv (bool): If True, save the ranking to `csv_path`.
        csv_path (str): Path to write CSV results when save_csv is True.

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'information_gain'] sorted by information_gain descending.
    """
    if logger:
        logger.info(f"Starting feature ranking (bins={bins})")

    if label_col is None:
        label_col = choose_label_column(df)
    if logger:
        logger.info(f"Using label column: {label_col}")
    # Use the label column and map it to a binary problem: 'BENIGN' vs 'ATTACK' (all non-BENIGN labels counted as attacks)
    y = df[label_col].astype(str)
    y = y.map(lambda v: 'BENIGN' if v == 'BENIGN' else 'ATTACK').astype('category')
    features = [c for c in df.columns if c != label_col]
    rows = []
    for f in features:
        ig = information_gain(df[f], y, bins=bins)
        rows.append({'feature': f, 'information_gain': ig})
    res = pd.DataFrame(rows).sort_values('information_gain', ascending=False).reset_index(drop=True)
    if save_csv:
        res.to_csv(csv_path, index=False)
        if logger:
            logger.info(f"Feature ranking saved to: {csv_path}")
    return res
