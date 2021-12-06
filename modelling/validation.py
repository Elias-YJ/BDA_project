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


def accuracy_df(file_path, cutoff_prob=0.5):
    s = stan.from_csv(file_path)
    prob = expit(s.stan_variable('theta'))
    pred = (prob > cutoff_prob).astype(int)

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
    stat['acc'] = stat['tp'] + stat['tn']

    return pd.DataFrame(data=stat)


def create_accuracy_data(prob=0.5):
    output = {}
    model_types = ['logreg', 'hier']
    for model_type in model_types:
        output[model_type] = accuracy_df(f'inference/{model_type}/*[1-4].csv', cutoff_prob=prob)
    return output


def plot_accuracy(prob=0.5):
    model_types = ['logreg', 'hier']
    for model_type in model_types:
        df = accuracy_df(f'inference/{model_type}/*[1-4].csv', cutoff_prob=prob)

        fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=100)
        sns.kdeplot(df['tp'], ax=ax[0, 0])
        sns.kdeplot(df['fp'], ax=ax[0, 1])
        sns.kdeplot(df['fn'], ax=ax[1, 0])
        sns.kdeplot(df['tn'], ax=ax[1, 1])

        ax[0, 0].set_xlabel('True positives / total')
        ax[0, 1].set_xlabel('False positives / total')
        ax[1, 0].set_xlabel('False negatives / total')
        ax[1, 1].set_xlabel('True negatives / total')

        ax[0, 0].set_title(f'Mean value: {np.mean(df["tp"]):.2f}')
        ax[0, 1].set_title(f'Mean value: {np.mean(df["fp"]):.2f}')
        ax[1, 0].set_title(f'Mean value: {np.mean(df["fn"]):.2f}')
        ax[1, 1].set_title(f'Mean value: {np.mean(df["tn"]):.2f}')

        fig.tight_layout(pad=3.0)

        plt.savefig(f'plots/{model_type}_conf_hist.pdf')
