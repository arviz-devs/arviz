data {
  int<lower=0> N;
  vector<lower=0>[N] time_since_joined;
  vector[N] slack_comments;
  vector[N] github_commits;
}

parameters {
  real b0;
  real b1;
  real log_b_sigma;

  real c0;
  real c1;
  real log_c_sigma;
}

transformed parameters {
  real<lower=0> b_sigma = exp(log_b_sigma);
  real<lower=0> c_sigma = exp(log_c_sigma);
}

model {
  b0 ~ normal(0,200);
  b1 ~ normal(0,200);
  b_sigma ~ normal(0,300);
  slack_comments ~ normal(b0 + b1 * time_since_joined, b_sigma);
  github_commits ~ normal(c0 + c1 * time_since_joined, c_sigma);

}

generated quantities {
    // elementwise log likelihood
    vector[N] log_likelihood_slack_comments;
    vector[N] log_likelihood_github_commits;

    // posterior predictive
    vector[N] slack_comments_hat;
    vector[N] github_commits_hat;

    //log likelihood & posterior predictive
    for (n in 1:N) {
        log_likelihood_slack_comments[n] = normal_lpdf(slack_comments[n] | b0 + b1 * time_since_joined[n], b_sigma);
        slack_comments_hat[n] = normal_rng(b0 + b1 * time_since_joined[n], b_sigma);

        log_likelihood_github_commits[n] = normal_lpdf(github_commits[n] | c0 + c1 * time_since_joined[n], c_sigma);
        github_commits_hat[n] = normal_rng(c0 + c1 * time_since_joined[n], c_sigma);
    }
}
