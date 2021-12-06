from typing import Dict
import cmdstanpy as stan
import pandas as pd
import numpy as np
import cmdstanpy as stan
# import matplotlib.pyplot as plt
from scipy import stats
from os.path import join
import pickle
import shutil


from modelling.util.data_loader import load_data
from modelling.util.data_formatters import format_logreg_data, format_hierarchical_data

STAN_OPTS = dict(
    show_progress=True,
    iter_sampling=4000,
    threads_per_chain=4,
    seed=1,
    adapt_delta=0.999,
    max_treedepth=12,
)

OUTPUT_DIR='./inference'

def run_models(stan_opts: Dict = None, output_dir: str = OUTPUT_DIR, **kwargs):
    run_list = kwargs.get('model_list', ['logreg', 'hier'])
    hier_column = kwargs.get('hier_column', 'chest_pain_type')
    df = load_data(include_dummies=True, norm_data=True)
    if 'logreg' in run_list:
        run_logreg(df, stan_opts, output_dir)
    if 'hier' in run_list:
        run_hier(df, stan_opts, output_dir, hier_column=hier_column)


def run_logreg(df, stan_opts, output_dir):
    model = stan.CmdStanModel(stan_file='modelling/stan_models/simple_regression.stan', model_name='logreg')
    data = format_logreg_data(df)
    priors = get_logreg_priors(data)
    run_model(model, data, priors, stan_opts, output_dir)


def run_hier(df, stan_opts, output_dir, hier_column):
    model = stan.CmdStanModel(stan_file='modelling/stan_models/hierarchical_v5.stan', model_name='hier')
    data = format_hierarchical_data(df, [hier_column])
    priors = get_hier_priors(data)
    run_model(model, data, priors, stan_opts, output_dir)


def get_logreg_priors(data):
    priors = dict(
        alpha_mu = 0,
        alpha_scale = 1,
        beta_mu = np.zeros(data['M'])+1,
        beta_scale = np.ones(data['M']),
    )
    return priors


def get_hier_priors(data):
    priors = dict(
        am_mu = 0,
        am_scale = 1,
        bm_mu = np.zeros(data['M']),
        bm_scale = np.ones(data['M']),
        as_mu = 0,
        as_scale = 1,
        bs_mu = np.zeros(data['M']),
        bs_scale = np.ones(data['M']),
    )
    return priors


def run_model(model: stan.CmdStanModel, data: Dict, priors: Dict, stan_opts: Dict = None, output_dir: str = OUTPUT_DIR):
    if stan_opts is None: stan_opts = STAN_OPTS
    dir = join(output_dir, model.name)
    shutil.rmtree(dir, ignore_errors=True)
    fit = model.sample(
        {**data, **priors},
        output_dir=dir,
        **stan_opts)
    summary = fit.summary()
    with open(join(dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    with open(join(dir, 'priors.pkl'), 'wb') as f:
        pickle.dump(priors, f)
    short_summary = summary.filter(regex=r'(alpha|beta)', axis=0)
    short_summary.to_csv(join(dir, 'summary.csv'))
    print(short_summary)