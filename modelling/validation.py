def cross_validate():
    import arviz as az
    model_types = ['logreg', 'hier']
    loos = []

    for model_type in model_types:
        file_path = f'inference/{model_type}/*[1-4].csv'

        # Load model data from sampling output files
        model = az.from_cmdstan(file_path, log_likelihood='log_lik')
        loo = az.loo(model, pointwise=True)
        loo.to_csv(f'inference/{model_type}_loo.csv')
        loos.append(loo)

    return loos
