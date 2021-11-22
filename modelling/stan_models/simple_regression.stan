data {
   int N;           // count of observations
   int M;           // count of regressors
   int y[N];        // training labels
   matrix[N, M] X;  // regressors
   real alpha_mu;
   real<lower=0> alpha_scale;
   vector[M] beta_mu;
   vector<lower=0>[M] beta_scale; 
}
parameters {
   vector[M] beta;  // weights for regressors
   real alpha;      // constant
}
model {
   alpha ~ normal(alpha_mu, alpha_scale);
   beta ~ normal(beta_mu, beta_scale);
   y ~ bernoulli_logit_glm(X, alpha, beta);
}
generated quantities {
    vector[N] log_lik;
    for (n in 1:N){
        log_lik[n] = bernoulli_logit_lpmf(a[n])
    }
}