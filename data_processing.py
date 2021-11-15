import pandas as pd


def add_split_columns(df):
    return


def add_mapped_columns(df):
    return


def process(raw='data/heart.csv', output='data/model_input.csv'):
    raw_data = pd.read_csv(raw)
    data = raw_data.copy()
    data = add_split_columns(data)
    data = add_mapped_columns(data)
    data.to_csv(output)
