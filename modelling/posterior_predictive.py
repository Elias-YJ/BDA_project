import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmdstanpy as stan
from modelling.util.data_loader import load_data

def points_to_plot(y, theta_hat):
    parts = 10
    df = pd.DataFrame(dict(y=y, theta=theta_hat))
    df = df.sort_values(by='theta')
    summary = []
    ps = np.linspace(0, 1, parts + 1)
    for i in np.arange(parts):
        low, high = ps[[i, i+1]]
        df_slice = df[df.theta.between(low, high)]
        summary.append(pd.Series(dict(
            mean_theta = df_slice.theta.mean(),
            mean_y = df_slice.y.mean(),
            sem_y = df_slice.y.std() / df_slice.shape[0] ** 0.5,
        )))
    summary = pd.concat(summary, axis=1).transpose()

    fig, ax = plt.subplots(figsize=(6, 6))
    y_offsets = np.random.normal(0, scale=0.005, size=df.shape[0])
    ax.scatter(df.theta, df.y + y_offsets, s=8, alpha=1, color='#000')
    ax.plot([0, 1], [0, 1], linestyle='--', dashes=(5, 5), linewidth=1, color='#666')
    ax.errorbar(summary.mean_theta, summary.mean_y, summary.sem_y, capsize=5, color='#000', fmt='o')
    return fig, ax


def make_plots():
    model_logreg = stan.from_csv('inference/logreg/*[1-4].csv')
    model_hier = stan.from_csv('inference/hier/*[1-4].csv')
    y = load_data()['heart_disease'].values

    model_names = ['logistic regression', 'hierarchical']
    fig_names = ['logreg', 'hier']
    models = [model_logreg, model_hier]

    def inv_logit(t):
        return 1/(1 + np.exp(-t))

    for model, name, figname in zip(models, model_names, fig_names):
        theta_hat = inv_logit(model.stan_variable('theta')).mean(axis=0)
        fig, ax = points_to_plot(y, theta_hat)
        ax.set_title(f'Posterior predictive: {name} model')
        ax.set_xlabel(r'mean $\theta$')
        ax.set_ylabel('y')
        fig.savefig(f'plots/post_pred_{figname}.pdf')