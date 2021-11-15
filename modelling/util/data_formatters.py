from .data_loader import load_data
import pandas as pd

Y_NAME = 'heart_disease'


def format_logreg_data(df):
    y = df[Y_NAME]

    df = df.drop(columns=[Y_NAME])
    df_numeric = df.select_dtypes(include=['int64','float64'])
    #print(f'Selected columns {df_numeric.columns}')
    X = df_numeric
    N, M = X.shape
    data = dict(
        N=N,
        M=M,
        y=y,
        X=X,
    )
    return data


def load_logreg_data():
    """Returns a dict containing the data for simple_regeression model"""
    df = load_data()
    return format_logreg_data(df)



    