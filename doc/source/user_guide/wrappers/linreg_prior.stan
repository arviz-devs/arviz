data {
  int<lower=0> N;
  real time_since_joined[N];
}

generated quantities {
    real b0;
    real b1;
    real log_b_sigma;
    real<lower=0> b_sigma;

    real c0;
    real c1;
    real log_c_sigma;
    real<lower=0> c_sigma;

    vector[N] slack_comments_hat;
    vector[N] github_commits_hat;

    b0 = normal_rng(0,200);
    b1 = normal_rng(0,200);
    b_sigma = abs(normal_rng(0,300));
    log_b_sigma = log(b_sigma);

    c0 = normal_rng(0,10);
    c1 = normal_rng(0,10);
    c_sigma = fabs(normal_rng(0,6));
    log_c_sigma = log(b_sigma);

    for (n in 1:N) {
        slack_comments_hat[n] = normal_rng(b0 + b1 * time_since_joined[n], b_sigma);
        github_commits_hat[n] = normal_rng(c0 + c1 * time_since_joined[n], c_sigma);
    }
}
