data {
    int N;                                  // count of observations
    int M;                                  // count of regressors
    int y[N];                               // training labels
    matrix[N, M] X;                         // regressors
    int<lower=1> J;                         // count of hierarchical groups (level 1)
    array[N] int<lower=1, upper=J> gj;      // count of datapoints in group j
    real<lower=0> as_alpha;
    real<lower=0> as_beta;
    real<lower=0> bs_alpha;
    real<lower=0> bs_beta;
    real<lower=0> am_scale;
    real<lower=0> bm_scale;
    real<lower=0> as_rate;
    real<lower=0> bs_rate;
}
// transformed data {
//     array[J] int start_index;
//     array[J] int end_index;
//     for (j in 1:J) {
//         start_index[j] = sum(Nj[1:j]) + 1;
//         end_index[j] = start_index[j] + Nj[j];
//     }
// }
parameters {
    real alpha_mu;             // hyper params
    real<lower=0> alpha_sigma;          //
    vector[M] beta_mu;         //
    vector<lower=0>[M] beta_sigma;      //

    array[J] vector[M] beta;   // weights for regressors
    array[J] real alpha;       // constant
}
model {
    alpha_mu ~ normal(0, am_scale);
    beta_mu ~ normal(0, bm_scale);
    // alpha_sigma ~ inv_gamma(as_alpha, as_beta);
    // beta_sigma ~ inv_gamma(bs_alpha, bs_beta);
    alpha_sigma ~ cauchy(0, as_rate);
    beta_sigma ~ cauchy(0, bs_rate);
    for (j in 1:J) {
        alpha[j] ~ normal(alpha_mu, alpha_sigma);
        beta[j] ~ normal(beta_mu, beta_sigma);     // todo: tää saattaa castata nää tyypit väärin. Pitää ehdä vaan tehdä for-looppi tai sit muuttaa tyypit jotenkin
        //y[start_index[j]:end_index[j]] ~ bernoulli_logit_glm(X[start_index[j]:end_index[j], :], alpha[j], beta[j]);
    }

    for (i in 1:N)
        y ~ bernoulli_logit_glm(X[i, :], alpha[gj[i]], beta[gj[i]]);
}