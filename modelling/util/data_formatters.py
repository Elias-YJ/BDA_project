from .data_loader import drop_dummy_cols
import pandas as pd

Y_NAME = 'heart_disease'


def format_logreg_data(df: pd.DataFrame):
    """
    returns:
    dict with
        - y: 'heart_disease' column
        - X: numeric values other than y
        - N: number of rows
        - M: number of columns in X
    """
    y = df[Y_NAME]

    df = df.drop(columns=[Y_NAME])
    df_numeric = df.select_dtypes(include=['int64','float64'])
    X = df_numeric
    N, M = X.shape
    data = dict(
        N=N,
        M=M,
        y=y,
        X=X,
    )
    return data


def format_hierarchical_data(df, grouping_cols):
    """
    parameters:
        - df: data df with normalizations applied
        - grouping_cols: list of columns to be used in grouping

    returns:
    dict with
        - y: 'heart_disease' column
        - X: numeric values other than y, grouping_cols, and
             dummy values related to grouping cols.
        - N: number of rows
        - M: number of columns in X
        - J: number of groups
        - gj: group each row belongs to
    """
    y_name = 'heart_disease'
    y = df[y_name]

    df = df.drop(columns=[y_name])
    df = drop_dummy_cols(df, grouping_cols)
    df_numeric = df.select_dtypes(include=['int64','float64'])
    df_numeric = df_numeric.drop(columns=grouping_cols, errors='ignore')
    group_indicies = df.groupby(grouping_cols).ngroup() + 1
    mapping_index = pd.MultiIndex.from_frame(df[grouping_cols])
    group_indicies.index = mapping_index
    X = df_numeric
    N, M = X.shape
    data = dict(
        N=N,
        M=M,
        y=y,
        X=X,
        J=group_indicies.max(),
        gj=group_indicies,
    )
    return data

def get_index_mapper(gj):
    """Convert gj to index mapping"""
    mapping = gj.drop_duplicates().sort_values()
    mapping = mapping.reset_index().set_index(0)
    mapping.index.name = 'j'
    return mapping
    