import cmdstanpy as stan
import pandas as pd
import numpy as np
import cmdstanpy as stan
# import matplotlib.pyplot as plt
from scipy import stats
from os.path import join
import pickle

from modelling.util.data_loader import load_data
from modelling.util.data_formatters import format_logreg_data, format_hierarchical_data


def run_all():
    stan_opts = dict(
        show_progress=True,
        output_dir='./inference',
        iter_sampling=4000,
        threads_per_chain=4,
        seed=1
    )


    df = load_data(include_dummies=True, norm_data=True)

    logreg_data = format_logreg_data(df)
    logreg_priors = dict(
        alpha_mu = 0,
        alpha_scale = 1,
        beta_mu = np.zeros(logreg_data['M']),
        beta_scale = np.ones(logreg_data['M']),
    )
    logreg_input = {**logreg_data, **logreg_priors}


    hier_data = format_hierarchical_data(df, ['chest_pain_type'])
    hier_priors = dict(
        am_mu = 0,
        am_scale = 1,
        bm_mu = np.zeros(hier_data['M']),
        bm_scale = np.ones(hier_data['M']),
        as_mu = 0,
        as_scale = 1,
        bs_mu = np.zeros(hier_data['M']),
        bs_scale = np.ones(hier_data['M']),
    )
    hier_input = {**hier_data, **hier_priors}


    logreg_model = stan.CmdStanModel(stan_file='modelling/stan_models/simple_regression.stan')
    logreg_fit = logreg_model.sample(logreg_input, **stan_opts)
    logreg_summary = logreg_fit.summary()


    hier_model = stan.CmdStanModel(stan_file='modelling/stan_models/hierarchical_v5.stan')
    hier_fit = hier_model.sample(hier_input, **stan_opts)
    hier_summary = hier_fit.summary()

    with open(join(stan_opts['output_dir'], 'logreg_input.pkl'), 'wb') as f:
        pickle.dump(logreg_input, f)

    with open(join(stan_opts['output_dir'], 'hier_input.pkl'), 'wb') as f:
        pickle.dump(hier_input, f)

    short_logreg_summary = logreg_summary.filter(regex=r'(alpha|beta)', axis=0)
    short_hier_summary = hier_summary.filter(regex=r'(alpha|beta)\[', axis=0)
    print(short_logreg_summary)
    print(short_hier_summary)

    short_logreg_summary.to_csv(join(stan_opts['output_dir'], 'logreg_summary.csv'))
    short_hier_summary.to_csv(join(stan_opts['output_dir'], 'hier_summary.csv'))
