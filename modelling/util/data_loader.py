import pandas as pd
import re


def format_names(df):
    """Rename columns from CamelCase to snake_case"""
    pattern = re.compile(r'[a-z][A-Z]')

    def add_underscore(m):
        return f'{m[0][0]}_{m[0][1]}'

    def format_name(name):
        return re.sub(pattern, add_underscore, name).lower()
    
    return df.columns.to_series().apply(format_name)


def add_split_columns(df):
    df = df.copy()
    df['RestingECG_ST'] = (df['RestingECG'] == 'ST').astype(int)
    df['RestingECG_LVH'] = (df['RestingECG'] == 'LVH').astype(int)

    df['ChestPainATA'] = (df['ChestPainType'] == 'ATA').astype(int)
    df['ChestPainNAP'] = (df['ChestPainType'] == 'NAP').astype(int)
    df['ChestPainASY'] = (df['ChestPainType'] == 'ASY').astype(int)
    df['ChestPainTA'] = (df['ChestPainType'] == 'TA').astype(int)

    df['ST_up'] = (df['ST_Slope'] == 'Up').astype(int)
    df['ST_flat'] = (df['ST_Slope'] == 'Flat').astype(int)
    df['ST_down'] = (df['ST_Slope'] == 'Down').astype(int)
    return df


def load_data(path='data/heart.csv'):
    """import this and call it to load the dataframe"""
    df = pd.read_csv(path)
    df = add_split_columns(df)
    df.columns = format_names(df)

    df.exercise_angina = df.exercise_angina.map({'Y': 1, 'N': 0})
    df.sex = df.sex.map({'M': 1, 'F': 0})
    
    return df