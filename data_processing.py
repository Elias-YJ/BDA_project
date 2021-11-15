import pandas as pd


def add_split_columns(df):
    df = df.copy()
    df['RestingECG_ST'] = (df['RestingECG'] == 'ST').astype(int)
    df['RestingECG_LVH'] = (df['RestingECG'] == 'LVH').astype(int)

    df['ChestPainATA'] = (df['ChestPainType'] == 'ATA').astype(int)
    df['ChestPainNAP'] = (df['ChestPainType'] == 'NAP').astype(int)
    df['ChestPainASY'] = (df['ChestPainType'] == 'ASY').astype(int)
    df['ChestPainTA'] = (df['ChestPainType'] == 'TA').astype(int)
    return df


def add_mapped_columns(df):
    df = df.copy()
    st_map = {'Up': 1, 'Flat': 0, 'Down': -1}
    angina_map = {'Y': 1, 'N': 0}
    gender_map = {'F': 0, 'M': 1}

    df['ST_Slope'] = df['ST_Slope'].map(st_map)
    df['ExerciseAngina'] = df['ExerciseAngina'].map(angina_map)
    df['Gender'] = df['Sex'].map(gender_map)
    return df


def drop_unused_columns(df):
    df = df.copy()
    df = df.drop(['RestingECG', 'ChestPainType', 'Sex'], axis=1)
    return df


def process(raw='data/heart.csv', output='data/model_input.csv'):
    raw_data = pd.read_csv(raw)
    data = raw_data.copy()
    data = add_split_columns(data)
    data = add_mapped_columns(data)
    data = drop_unused_columns(data)
    data.to_csv(output)
    return data
