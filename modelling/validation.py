import pandas as pd
import numpy as np
from scipy.special import expit
from .util import data_loader
import arviz as az
import cmdstanpy as stan
import seaborn as sns
import matplotlib.pyplot as plt

def cross_validate(dir = 'inference'):
    model_types = ['logreg', 'hier']
    loos = []

    for model_type in model_types:
        file_path = f'{dir}/{model_type}/*[1-4].csv'

        # Load model data from sampling output files
        model = az.from_cmdstan(file_path, log_likelihood='log_lik')
        loo = az.loo(model, pointwise=True)
        loo.to_csv(f'{dir}/{model_type}_loo.csv')
        loos.append(loo)

    return loos


def accuracy_df(file_path):
    s = stan.from_csv(file_path)
    prob = expit(s.stan_variable('theta'))
    pred = (prob > 0.5).astype(int)

    y = data_loader.load_data().loc[:, 'heart_disease'].values

    stat = {
        'tp': y & pred,
        'fp': pred > y,
        'tn': (1 - pred) & (1 - y),
        'fn': y > pred
    }

    for key, value in stat.items():
        stat[key] = np.sum(value, axis=1) / len(y)

    stat['sens'] = stat['tp']/(stat['tp']+stat['fn'])
    stat['spec'] = stat['tn'] / (stat['tn'] + stat['fp'])

    return pd.DataFrame(data=stat)


def plot_accuracy():
    model_types = ['logreg', 'hier']
    for model_type in model_types:
        df = accuracy_df(f'inference/{model_type}/*[1-4].csv')

        fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=120)
        sns.histplot(df['tp'], ax=ax[0, 0])
        sns.histplot(df['fp'], ax=ax[0, 1])
        sns.histplot(df['tn'], ax=ax[1, 0])
        sns.histplot(df['fp'], ax=ax[1, 1])
        plt.savefig(f'figures/{model_type}_conf_hist.pdf')