data {
    int N;                                  // count of observations
    int M;                                  // count of regressors
    array[N] int y;                               // training labels
    matrix[N, M] X;                         // regressors
    int<lower=1> J;                         // count of hierarchical groups (level 1)
    array[N] int<lower=1, upper=J> gj;      // count of datapoints in group j
    real                am_mu;
    real<lower=0>       am_scale;
    vector[M]           bm_mu;
    vector<lower=0>[M]  bm_scale;
    real                as_mu;
    real<lower=0>       as_scale;
    real                bs_mu;
    real<lower=0>       bs_scale;

}
parameters {
    real                alpha_mu;
    real<lower=0>       alpha_sigma;
    vector[M]           beta_mu;
    real<lower=0>       beta_sigma;

    array[J] real       alpha;
    array[J] vector[M]  beta;
}
model {
    alpha_mu ~ normal(am_mu, am_scale);
    beta_mu ~ normal(bm_mu, bm_scale);
    alpha_sigma ~ normal(as_mu, as_scale);
    beta_sigma ~ normal(bs_mu, bs_scale);

    alpha ~ normal(alpha_mu, alpha_sigma);
    for (j in 1:J)
        beta[j] ~ normal(beta_mu, beta_sigma);
    {
        vector[N] a;
        for (i in 1:N) {
            a[i] = alpha[gj[i]] + X[i, :] * beta[gj[i]];
        }
        // y ~ bernoulli_logit_glm(X, a, b);
        y ~ bernoulli_logit(a);
    }
    // for (i in 1:N)
    //     y[i] ~ bernoulli_logit_glm(X[i, :], alpha[gj[i]], beta[gj[i]]);
}