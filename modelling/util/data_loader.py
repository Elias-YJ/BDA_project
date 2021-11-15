import pandas as pd
import re

ENCODABLE_COLUMNS = ['resting_ecg', 'chest_pain_type', 'st_slope']

def format_names(df):
    """Rename columns from CamelCase to snake_case"""
    pattern = re.compile(r'[a-z][A-Z]')

    def add_underscore(m):
        return f'{m[0][0]}_{m[0][1]}'

    def format_name(name):
        return re.sub(pattern, add_underscore, name).lower()
    
    return df.columns.to_series().apply(format_name)


def add_split_columns(df, effect_coding=False):
    df = df.copy()
    # df['RestingECG_ST'] = (df['RestingECG'] == 'ST').astype(int)
    # df['RestingECG_LVH'] = (df['RestingECG'] == 'LVH').astype(int)

    # df['ChestPainATA'] = (df['ChestPainType'] == 'ATA').astype(int)
    # df['ChestPainNAP'] = (df['ChestPainType'] == 'NAP').astype(int)
    # df['ChestPainASY'] = (df['ChestPainType'] == 'ASY').astype(int)
    # df['ChestPainTA'] = (df['ChestPainType'] == 'TA').astype(int)

    # df['ST_up'] = (df['ST_Slope'] == 'Up').astype(int)
    # df['ST_flat'] = (df['ST_Slope'] == 'Flat').astype(int)
    # df['ST_down'] = (df['ST_Slope'] == 'Down').astype(int)

    for col in ENCODABLE_COLUMNS:
        for val in df[col].unique():
            mask = (df[col] == val)
            col_val = f'{col}_{val.lower()}'
            df[col_val] = mask.astype(int)
            if effect_coding:
                df.loc[~mask, col_val] = -1
    return df


def drop_dummy_cols(df, drop_cols = None):
    if not drop_cols:
        drop_cols = ENCODABLE_COLUMNS
    df = df.copy()
    for col in drop_cols:
        name_mask = df.columns.str.startswith(col)
        equal_mask = df.columns != col
        drop_cols = df.columns[name_mask & equal_mask]
        df = df.drop(columns=drop_cols, errors='ignore')
    return df

def normalize(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=['int64','float64'])
    zero_one_cols = num_cols.columns[num_cols.isin([0, 1]).all()]
    num_cols = num_cols.drop(columns=zero_one_cols, errors='ignore')
    num_cols -= num_cols.mean()
    num_cols /= num_cols.std()
    return pd.concat([num_cols, df.drop(columns=num_cols.columns)], axis=1)


def load_data(path='data/heart.csv', include_dummies=False, effect_coding=False, norm_data=False):
    """import this and call it to load the dataframe"""
    df = pd.read_csv(path)
    df.columns = format_names(df)
    if norm_data:
        df = normalize(df)
    if include_dummies:
        df = add_split_columns(df, effect_coding)
    
    df.exercise_angina = df.exercise_angina.map({'Y': 1, 'N': 0})
    df.sex = df.sex.map({'M': 1, 'F': 0})
    
    return df