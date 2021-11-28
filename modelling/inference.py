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
    seed=1
)

# OUTPUT_DIR='./inference'
OUTPUT_DIR='./inference_sex'

def run_models(stan_opts: Dict = None, output_dir: str = OUTPUT_DIR):
    # df = load_data(include_dummies=True, norm_data=True)

    # logreg_data = format_logreg_data(df)
    # logreg_input = {**logreg_data, **logreg_priors}


    # hier_data = format_hierarchical_data(df, ['chest_pain_type'])
    # hier_priors = dict(
    #     am_mu = 0,
    #     am_scale = 1,
    #     bm_mu = np.zeros(hier_data['M']),
    #     bm_scale = np.ones(hier_data['M']),
    #     as_mu = 0,
    #     as_scale = 1,
    #     bs_mu = np.zeros(hier_data['M']),
    #     bs_scale = np.ones(hier_data['M']),
    # )
    # hier_input = {**hier_data, **hier_priors}


    # logreg_model = stan.CmdStanModel(stan_file='modelling/stan_models/simple_regression.stan')
    # hier_model = stan.CmdStanModel(stan_file='modelling/stan_models/hierarchical_v5.stan')

    # shutil.rmtree(output_dir)
    
    # logreg_fit = logreg_model.sample(
    #     logreg_input,
    #     output_dir = join(output_dir, 'logreg'),
    #     **stan_opts)
    # logreg_summary = logreg_fit.summary()


    # hier_fit = hier_model.sample(
    #     hier_input,
    #     output_dir = join(output_dir, 'hier'),
    #     **stan_opts)
    # hier_summary = hier_fit.summary()

    # with open(join(output_dir, 'logreg', 'logreg_input.pkl'), 'wb') as f:
    #     pickle.dump(logreg_input, f)

    # with open(join(output_dir, 'hier', 'hier_input.pkl'), 'wb') as f:
    #     pickle.dump(hier_input, f)

    # short_logreg_summary = logreg_summary.filter(regex=r'(alpha|beta)', axis=0)
    # short_hier_summary = hier_summary.filter(regex=r'(alpha|beta)\[', axis=0)
    # print(short_logreg_summary)
    # print(short_hier_summary)

    # short_logreg_summary.to_csv(join(output_dir, 'logreg', 'logreg_summary.csv'))
    # short_hier_summary.to_csv(join(output_dir, 'hier', 'hier_summary.csv'))

    logreg_model = stan.CmdStanModel(stan_file='modelling/stan_models/simple_regression.stan', model_name='logreg')
    hier_model = stan.CmdStanModel(stan_file='modelling/stan_models/hierarchical_v5.stan', model_name='hier')
    df = load_data(include_dummies=True, norm_data=True)

    models = [logreg_model, hier_model]
    datasets = [
        format_logreg_data(df),
        # format_hierarchical_data(df, ['chest_pain_type'])
        format_hierarchical_data(df, ['sex'])
    ]
    prior_formatters = [get_logreg_priors, get_hier_priors]
    priors = [fmt(data) for fmt, data in zip(prior_formatters, datasets)]

    for model, data, priors in zip(models, datasets, priors):
        run_model(model, data, priors, stan_opts=stan_opts, output_dir=output_dir)



def get_logreg_priors(data):
    priors = dict(
        alpha_mu = 0,
        alpha_scale = 1,
        beta_mu = np.zeros(data['M']),
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