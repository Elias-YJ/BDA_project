data {
   int N;           // count of observations
   int M;           // count of regressors
   int y[N];        // training labels
   matrix[N, M] X;  // regressors
}
parameters {
   vector[M] beta;  // weights for regressors
   real alpha;      // constant
}
model {   
   y ~ bernoulli_logit_glm(X, alpha, beta);
}
generated quantities {
   vector[N] theta = logit(alpha + X * beta);
}