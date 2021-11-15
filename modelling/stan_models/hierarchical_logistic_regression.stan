data {
    int N;                           // count of observations
    int M;                           // count of regressors
    int y[N];                        // training labels
    matrix[N, M] X;                  // regressors
    int<lower=1> J;                  // count of hierarchical groups (level 1)
    array[J] int<lower=0> Nj;        // count of datapoints in group j
}
transformed data {
    array[J] int start_index;
    array[J] int end_index;
    for (j in 1:J) {
        start_index[j] = sum(Nj[1:j]) + 1;
        end_index[j] = start_index[j] + Nj[j];
    }
}
parameters {
    real alpha_mu;             // hyper params
    real alpha_sigma;          //
    vector[M] beta_mu;         //
    vector[M] beta_sigma;      //

    array[J] vector[M] beta;   // weights for regressors
    array[J] real alpha;       // constant
}
model {
    for (j in 1:J) {
        alpha[j] ~ normal(alpha_mu, alpha_sigma);
        beta[j] ~ normal(beta_mu, beta_sigma);     // todo: tää saattaa castata nää tyypit väärin. Pitää ehdä vaan tehdä for-looppi tai sit muuttaa tyypit jotenkin
        y[start_index[j]:end_index[j]] ~ bernoulli_logit_glm(X[start_index[j]:end_index[j], :], alpha[j], beta[j]);
    }
}