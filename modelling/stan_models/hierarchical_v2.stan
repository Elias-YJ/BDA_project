data {
    int N;                                  // count of observations
    int M;                                  // count of regressors
    int y[N];                               // training labels
    matrix[N, M] X;                         // regressors
    int<lower=1> J;                         // count of hierarchical groups (level 1)
    array[N] int<lower=1, upper=J> gj;      // count of datapoints in group j
    real                am_mu;
    real<lower=0>       am_scale;
    vector[M]           bm_mu;
    vector<lower=0>[M]  bm_scale;
    real                as_mu;
    real<lower=0>       as_scale;
    vector[M]           bs_mu;
    vector<lower=0>[M]  bs_scale;

}
parameters {
    real                alpha_mu;
    real<lower=0>       alpha_sigma;
    vector[M]           beta_mu;
    vector<lower=0>[M]  beta_sigma;

    array[J] real       alpha;
    array[J] vector[M]  beta;
}
model {
    alpha_mu ~ normal(am_mu, am_scale);
    beta_mu ~ normal(bm_mu, bm_scale);
    alpha_sigma ~ normal(as_mu, as_scale);
    beta_sigma ~ normal(bs_mu, bs_scale);
    for (j in 1:J) {
        alpha[j] ~ normal(alpha_mu, alpha_sigma);
        beta[j] ~ normal(beta_mu, beta_sigma);     // todo: tää saattaa castata nää tyypit väärin. Pitää ehdä vaan tehdä for-looppi tai sit muuttaa tyypit jotenkin
        //y[start_index[j]:end_index[j]] ~ bernoulli_logit_glm(X[start_index[j]:end_index[j], :], alpha[j], beta[j]);
    }

    for (i in 1:N)
        y ~ bernoulli_logit_glm(X[i, :], alpha[gj[i]], beta[gj[i]]);
}