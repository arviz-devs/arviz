data {
  // Define data for fitting
  int<lower=0> N;
  array[N] real x;
  array[N] real y;
  // Define excluded data. It will not be used when fitting.
  int<lower=0> N_ex;
  array[N_ex] real x_ex;
  array[N_ex] real y_ex;
}

parameters {
  real b0;
  real b1;
  real<lower=0> sigma_e;
}

model {
  b0 ~ normal(0, 10);
  b1 ~ normal(0, 10);
  sigma_e ~ normal(0, 10);
  for (i in 1:N) {
    y[i] ~ normal(b0 + b1 * x[i], sigma_e);  // use only data for fitting
  }

}

generated quantities {
    array[N] real log_lik;
    array[N_ex] real log_lik_ex;
    array[N] real y_hat;

    for (i in 1:N) {
        // calculate log likelihood and posterior predictive, there are
        // no restrictions on adding more generated quantities
        log_lik[i] = normal_lpdf(y[i] | b0 + b1 * x[i], sigma_e);
        y_hat[i] = normal_rng(b0 + b1 * x[i], sigma_e);
    }
    for (j in 1:N_ex) {
        // calculate the log likelihood of the excluded data given data_for_fitting
        log_lik_ex[j] = normal_lpdf(y_ex[j] | b0 + b1 * x_ex[j], sigma_e);
    }
}
