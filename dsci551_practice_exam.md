---
editor_options: 
  markdown: 
    wrap: 72
---

# DSCI 551: Mock Exam

## Lectures 5-8: Continuous Distributions, MLE, and Simulation

**Total Points**: 100

------------------------------------------------------------------------

## Table of Contents

| Section | Topic | Questions | Points |
|-----------------|-----------------|---------------------|-----------------|
| [Part A](#part-a-conceptual-questions) | Conceptual Questions | 1-10 | 20 |
| [Part B](#part-b-continuous-distributions) | Continuous Distributions | 11-15 | 20 |
| [Part C](#part-c-maximum-likelihood-estimation) | Maximum Likelihood Estimation | 16-20 | 25 |
| [Part D](#part-d-simulation-and-random-sampling) | Simulation and Random Sampling | 21-25 | 25 |
| [Part E](#part-e-integrated-problems) | Integrated Problems | 26-27 | 10 |
| [Answer Key](#answer-key) | Solutions | All | \- |

------------------------------------------------------------------------

## Part A: Conceptual Questions {#part-a-conceptual-questions}

**[20 points total - 2 points each]**

### Question 1

Explain the key difference between a probability mass function (PMF) and
a probability density function (PDF). Why is P(X = x) = 0 for continuous
random variables?

### Question 2

What are the two essential properties that define a valid probability
density function f_X(x)? Provide the mathematical conditions.

### Question 3

A colleague claims that a function cannot be a valid PDF if f_X(x) \> 1
for some values of x. Is this claim correct? Justify your answer with an
explanation of what PDF values represent.

### Question 4

Define what it means for a random sample to be "iid" (independent and
identically distributed). Give one example where independence might be
violated in practice.

### Question 5

Explain the concept of maximum likelihood estimation (MLE) in your own
words. What does it mean for a parameter estimate to "maximize the
likelihood"?

### Question 6

Why do we typically work with the log-likelihood function rather than
the likelihood function when performing MLE? List two specific
advantages.

### Question 7

What is the difference between the cumulative distribution function
F_X(x) and the survival function S_X(x)? Write the mathematical
relationship between them.

### Question 8

Explain what a "seed" is in the context of random number generation and
why setting a seed is important for reproducible data science work.

### Question 9

For a right-skewed distribution, what is the typical ordering of the
mode, median, and mean? Draw a simple sketch to illustrate.

### Question 10

If X and Y are independent continuous random variables, what is the
relationship between their joint PDF f\_{X,Y}(x,y) and their marginal
PDFs f_X(x) and f_Y(y)?

------------------------------------------------------------------------

## Part B: Continuous Distributions {#part-b-continuous-distributions}

**[20 points total]**

### Question 11 [4 points]

Suppose the time (in hours) until a server fails follows an Exponential
distribution with rate parameter λ = 0.2 failures per hour.

**(a)** [1 point] What is the mean time until failure?

**(b)** [2 points] Write R code to calculate the probability that the
server lasts more than 8 hours.

**(c)** [1 point] What property makes the Exponential distribution
special for modeling waiting times?

### Question 12 [4 points]

A continuous random variable X has the following PDF:

f_X(x) = c · x² for 0 ≤ x ≤ 2, and 0 otherwise

**(a)** [2 points] Find the value of c that makes this a valid PDF. Show
your work.

**(b)** [2 points] Write R code that uses numerical integration
(`integrate()`) to verify your answer by checking that the total
probability equals 1.

### Question 13 [4 points]

Daily rainfall in Vancouver (in mm) can be modeled as X \~ Gamma(k = 2,
θ = 3).

**(a)** [1 point] What is the expected daily rainfall?

**(b)** [2 points] Write R code to find the 90th percentile of rainfall
(the value below which 90% of days fall).

**(c)** [1 point] Write R code to calculate P(5 \< X \< 10).

### Question 14 [4 points]

Exam scores are normally distributed with mean 72 and standard deviation
8.

**(a)** [2 points] Write R code to find the probability that a randomly
selected student scores between 65 and 85.

**(b)** [2 points] Write R code to find the 95% prediction interval for
exam scores (i.e., the interval containing the middle 95% of scores).

### Question 15 [4 points]

Consider the joint PDF:

f\_{X,Y}(x,y) = 6xy for 0 ≤ x ≤ 1, 0 ≤ y ≤ 1, x + y ≤ 1

**(a)** [2 points] Write R code to approximate P(X \> 0.3, Y \< 0.4)
using a double integral or numerical method.

**(b)** [2 points] Are X and Y independent? Justify your answer
mathematically.

------------------------------------------------------------------------

## Part C: Maximum Likelihood Estimation {#part-c-maximum-likelihood-estimation}

**[25 points total]**

### Question 16 [5 points]

Consider a random sample of size n = 5 from an Exponential(β)
distribution with observed values: {2.3, 5.1, 1.8, 3.4, 4.2}.

**(a)** [2 points] Write out the likelihood function L(β \| data) for
this sample (do not simplify).

**(b)** [2 points] Write out the log-likelihood function log L(β \|
data) and simplify it.

**(c)** [1 point] What is the MLE estimate β̂ for this data? (You may use
the formula from the notes.)

### Question 17 [6 points]

You collect a random sample of waiting times (in minutes) that you
believe follow an Exponential distribution with scale parameter β.

Your data: `wait_times <- c(3.2, 7.5, 2.1, 5.8, 4.3, 6.1, 1.9, 8.2)`

**(a)** [3 points] Write R code to implement the empirical MLE
approach: - Create a sequence of β values from 0.1 to 10 - Calculate the
log-likelihood for each β value - Find the β that maximizes the
log-likelihood

**(b)** [2 points] Write R code to plot the log-likelihood function
against the β values.

**(c)** [1 point] Compare your empirical MLE to the analytical MLE
(sample mean). Do they match?

### Question 18 [5 points]

Derive the analytical MLE for the parameter p in a Bernoulli(p)
distribution.

Given: Random sample Y₁, Y₂, ..., Yₙ where each Yᵢ ∈ {0, 1}

**(a)** [2 points] Write the likelihood function L(p \| y₁, ..., yₙ).

**(b)** [2 points] Write the log-likelihood and take its derivative with
respect to p.

**(c)** [1 point] Solve for p̂ by setting the derivative equal to zero.
What is the MLE?

### Question 19 [4 points]

You are modeling the number of customers arriving at a store per hour
using a Poisson(λ) distribution. You observe the following counts over 6
hours: {8, 12, 7, 10, 9, 11}.

**(a)** [2 points] What is the likelihood function for λ given this
data?

**(b)** [2 points] Write R code to find the MLE estimate of λ
empirically, testing values from 1 to 20.

### Question 20 [5 points]

Explain the difference between the empirical and analytical approaches
to MLE.

**(a)** [2 points] When would you prefer to use the empirical approach?

**(b)** [2 points] When would you prefer to use the analytical approach?

**(c)** [1 point] What are the advantages of each method?

------------------------------------------------------------------------

## Part D: Simulation and Random Sampling {#part-d-simulation-and-random-sampling}

**[25 points total]**

### Question 21 [5 points]

Consider rolling a fair six-sided die.

**(a)** [2 points] Write R code to simulate 1000 rolls of a fair die and
store the results in a variable called `rolls`.

**(b)** [2 points] Write R code to calculate the empirical probability
mass function (PMF) from your simulation using `table()` or
`janitor::tabyl()`.

**(c)** [1 point] Set a seed of 42 at the beginning of your code to make
it reproducible.

### Question 22 [6 points]

Simulate sampling from a Normal(μ = 50, σ² = 100) distribution to
demonstrate the Law of Large Numbers.

**(a)** [3 points] Write R code to: - Set seed to 123 - Generate samples
of sizes n = 10, 100, 1000, and 10000 - Calculate the sample mean for
each sample size

**(b)** [2 points] Create a vector showing how the sample mean changes
with sample size.

**(c)** [1 point] What theoretical value should the sample mean approach
as n increases?

### Question 23 [6 points]

You want to estimate properties of a Weibull(λ = 5, k = 2) distribution
through simulation.

**(a)** [2 points] Write R code to generate 10,000 random samples from
this distribution.

**(b)** [2 points] Calculate the empirical mean and variance from your
sample.

**(c)** [2 points] Write R code to calculate the empirical 75th
percentile (0.75-quantile) and compare it to the theoretical value using
`qweibull()`.

### Question 24 [4 points]

Consider a multi-step simulation: First, flip a fair coin. If heads,
draw from Normal(10, 4); if tails, draw from Normal(20, 9).

**(a)** [3 points] Write R code to simulate this process 5000 times and
store all the results.

**(b)** [1 point] Calculate the overall empirical mean of your simulated
values.

### Question 25 [4 points]

A port receives X \~ Poisson(λ = 4) ships per day. Each ship requires a
uniformly distributed time between 2 and 6 hours to unload.

Write R code to simulate one day and calculate the total unloading time
needed:

**(a)** [2 points] Generate the number of ships arriving (one draw from
Poisson).

**(b)** [2 points] For each ship, generate its unloading time and sum
them to get total time.

------------------------------------------------------------------------

## Part E: Integrated Problems {#part-e-integrated-problems}

**[10 points total]**

### Question 26 [5 points]

You collect a dataset of customer purchase amounts (in dollars) and
believe they follow a Log-Normal distribution. Your data:

``` r
purchases <- c(25, 45, 30, 120, 55, 80, 35, 200, 60, 90)
```

**(a)** [2 points] Explain why you might choose a Log-Normal
distribution instead of a Normal distribution for this data.

**(b)** [3 points] Write R code to: - Transform the data to find the MLE
estimates of μ and σ² (hint: take log of data) - Use these estimates to
create a Log-Normal distribution - Generate 1000 random samples from
your fitted distribution

### Question 27 [5 points]

Design a simulation study to verify that the sample mean is an unbiased
estimator of the population mean for an Exponential(β = 5) distribution.

Write complete R code that: - Sets a seed for reproducibility - Repeats
the following 1000 times: - Generate a sample of size 30 from
Exponential(β = 5) - Calculate and store the sample mean - Calculate the
mean of all 1000 sample means - Compare this to the true population mean
(β = 5)

------------------------------------------------------------------------

# Answer Key {#answer-key}

## Part A: Conceptual Questions

### Answer 1

**PMF vs PDF:** A PMF gives the probability of each discrete outcome
(P(X = x) \> 0), while a PDF gives the density of probability at each
point for continuous variables. For continuous variables, P(X = x) = 0
because there are infinitely many possible values, so the probability of
any exact single value is zero. Instead, we calculate probabilities over
intervals using integration: P(a ≤ X ≤ b) = ∫[a to b] f(x)dx.

### Answer 2

**Valid PDF properties:** 1. **Non-negative:** f_X(x) ≥ 0 for all x 2.
**Total probability = 1:** ∫\_{-∞}\^{∞} f_X(x)dx = 1

### Answer 3

**Claim is INCORRECT.** A PDF can have values greater than 1. The
density represents concentration of probability per unit, not
probability itself. What matters is that the total area under the PDF
curve equals 1. For example, Uniform(0, 0.5) has PDF f(x) = 2 for 0 ≤ x
≤ 0.5, which exceeds 1, but the total area is 2 × 0.5 = 1.

### Answer 4

**iid definition:** A random sample is independent and identically
distributed if: 1. Each pair of observations is independent (outcome of
one doesn't affect others) 2. Each observation comes from the same
distribution

**Example of violated independence:** Daily temperatures in Vancouver
are not independent because today's temperature is correlated with
yesterday's temperature. Stock prices on consecutive days also violate
independence due to autocorrelation.

### Answer 5

**MLE concept:** Maximum Likelihood Estimation finds parameter values
that make the observed data "most likely" to have occurred. Given data
and an assumed distribution family, MLE asks: "Which parameter values
would maximize the probability (likelihood) of observing exactly this
data?" The estimate that maximizes this likelihood is the MLE.

### Answer 6

**Why log-likelihood?** 1. **Converts products to sums:** Since
likelihood is a product of densities, log transforms it to a sum, making
calculations easier 2. **Numerical stability:** Prevents underflow
issues when multiplying many small probabilities 3. **Easier
differentiation:** Sums are easier to differentiate than products for
optimization 4. **Monotonicity:** Since log is monotone increasing,
maximizing log-likelihood is equivalent to maximizing likelihood

### Answer 7

**CDF vs Survival Function:** - **CDF:** F_X(x) = P(X ≤ x) = probability
of being at or below x - **Survival Function:** S_X(x) = P(X \> x) =
probability of exceeding x - **Relationship:** S_X(x) = 1 - F_X(x)

### Answer 8

**Seed:** A seed is the initial value used to start a pseudorandom
number generator sequence. Computers generate "random" numbers
deterministically using algorithms—setting the same seed produces the
same sequence of numbers. This is crucial for reproducibility in data
science: others can replicate your exact results by using the same seed
value.

### Answer 9

**Right-skewed distribution ordering:** Mode \< Median \< Mean

```         
        /\
       /  \___
      /       \____
     /             \______
    |  |    |        |
  Mode Med Mean
```

The tail extends to the right, pulling the mean in that direction. The
mode is at the peak, median in the middle, and mean is pulled toward the
long tail.

### Answer 10

**Independent random variables:** If X and Y are independent:
**f\_{X,Y}(x,y) = f_X(x) · f_Y(y)**

The joint PDF is the product of the marginal PDFs. Independence means
knowing X provides no information about Y.

------------------------------------------------------------------------

## Part B: Continuous Distributions

### Answer 11

**(a)** Mean = 1/λ = 1/0.2 = **5 hours**

**(b)**

``` r
# P(X > 8) = 1 - P(X ≤ 8)
prob <- 1 - pexp(8, rate = 0.2)
# OR using survival function directly
prob <- pexp(8, rate = 0.2, lower.tail = FALSE)
# Answer: 0.2019
```

**(c)** The Exponential distribution is **memoryless**: P(X \> s + t \|
X \> s) = P(X \> t). The probability of surviving an additional time t
is the same regardless of how long you've already survived.

### Answer 12

**(a)** For valid PDF: ∫₀² c·x² dx = 1

∫₀² c·x² dx = c[x³/3]₀² = c(8/3 - 0) = 8c/3 = 1

Therefore: **c = 3/8**

**(b)**

``` r
# Define the PDF
pdf_func <- function(x) {
  ifelse(x >= 0 & x <= 2, (3/8) * x^2, 0)
}

# Verify integration
result <- integrate(pdf_func, lower = 0, upper = 2)
result$value  # Should be 1 (or very close due to numerical precision)
```

### Answer 13

**(a)** E(X) = k·θ = 2 × 3 = **6 mm**

**(b)**

``` r
# 90th percentile
q90 <- qgamma(0.90, shape = 2, scale = 3)
# Answer: approximately 11.67 mm
```

**(c)**

``` r
# P(5 < X < 10)
prob <- pgamma(10, shape = 2, scale = 3) - pgamma(5, shape = 2, scale = 3)
# OR
prob <- diff(pgamma(c(5, 10), shape = 2, scale = 3))
# Answer: approximately 0.405
```

### Answer 14

**(a)**

``` r
# P(65 < X < 85)
prob <- pnorm(85, mean = 72, sd = 8) - pnorm(65, mean = 72, sd = 8)
# OR
prob <- diff(pnorm(c(65, 85), mean = 72, sd = 8))
# Answer: approximately 0.842
```

**(b)**

``` r
# 95% prediction interval: [Q(0.025), Q(0.975)]
lower <- qnorm(0.025, mean = 72, sd = 8)
upper <- qnorm(0.975, mean = 72, sd = 8)
interval <- c(lower, upper)
# Answer: approximately [56.32, 87.68]
```

### Answer 15

**(a)**

``` r
# Using numerical integration
library(cubature)

# Define joint PDF
f_xy <- function(xy) {
  x <- xy[1]
  y <- xy[2]
  if (x >= 0 & x <= 1 & y >= 0 & y <= 1 & (x + y) <= 1) {
    return(6 * x * y)
  } else {
    return(0)
  }
}

# Region: X > 0.3, Y < 0.4, and x + y ≤ 1
# Need to integrate carefully over valid region
result <- adaptIntegrate(f_xy, 
                         lowerLimit = c(0.3, 0), 
                         upperLimit = c(1, 0.4))
# Due to constraint x + y ≤ 1, need more careful setup
# Approximate answer: ~0.087
```

**(b)** **No, X and Y are NOT independent.** For independence, we would
need f\_{X,Y}(x,y) = f_X(x)·f_Y(y) for all x,y. However, the constraint
x + y ≤ 1 creates dependence—the valid range of Y depends on the value
of X. Additionally, if you calculate the marginal distributions, you'll
find their product doesn't equal the joint PDF.

------------------------------------------------------------------------

## Part C: Maximum Likelihood Estimation

### Answer 16

**(a)** Likelihood function:

```         
L(β | data) = ∏ᵢ₌₁⁵ (1/β)exp(-yᵢ/β)
           = (1/β)⁵ exp(-(2.3 + 5.1 + 1.8 + 3.4 + 4.2)/β)
           = (1/β⁵) exp(-16.8/β)
```

**(b)** Log-likelihood:

```         
log L(β | data) = log[(1/β⁵) exp(-16.8/β)]
                = -5log(β) - 16.8/β
```

**(c)** MLE for Exponential: **β̂ = sample mean = 16.8/5 = 3.36**

### Answer 17

**(a)**

``` r
wait_times <- c(3.2, 7.5, 2.1, 5.8, 4.3, 6.1, 1.9, 8.2)
n <- length(wait_times)
sum_y <- sum(wait_times)

# Empirical MLE approach
beta_values <- seq(0.1, 10, by = 0.01)
log_lik <- -n * log(beta_values) - sum_y / beta_values

# Find maximum
beta_mle_empirical <- beta_values[which.max(log_lik)]
beta_mle_empirical
```

**(b)**

``` r
plot(beta_values, log_lik, type = "l",
     xlab = "Beta", ylab = "Log-Likelihood",
     main = "Log-Likelihood Function for Exponential Distribution")
abline(v = beta_mle_empirical, col = "red", lty = 2)
```

**(c)**

``` r
# Analytical MLE
beta_mle_analytical <- mean(wait_times)

# Compare
cat("Empirical MLE:", beta_mle_empirical, "\n")
cat("Analytical MLE:", beta_mle_analytical, "\n")
# They should match (approximately 4.89)
```

### Answer 18

**(a)** Likelihood function:

```         
PMF: P(Y = y | p) = p^y (1-p)^(1-y)

L(p | y₁,...,yₙ) = ∏ᵢ₌₁ⁿ p^yᵢ (1-p)^(1-yᵢ)
                 = p^(Σyᵢ) (1-p)^(n - Σyᵢ)
```

**(b)** Log-likelihood and derivative:

```         
log L(p | data) = (Σyᵢ)log(p) + (n - Σyᵢ)log(1-p)

d/dp log L = (Σyᵢ)/p - (n - Σyᵢ)/(1-p)
```

**(c)** Solve for MLE:

```         
(Σyᵢ)/p̂ - (n - Σyᵢ)/(1-p̂) = 0
(Σyᵢ)(1-p̂) = p̂(n - Σyᵢ)
Σyᵢ - p̂Σyᵢ = np̂ - p̂Σyᵢ
Σyᵢ = np̂

p̂ = Σyᵢ/n = ȳ (sample proportion)
```

### Answer 19

**(a)** Likelihood function:

```         
Poisson PMF: P(Y = y | λ) = (λ^y exp(-λ))/y!

L(λ | data) = ∏ᵢ₌₁⁶ (λ^yᵢ exp(-λ))/yᵢ!
            = λ^(Σyᵢ) exp(-6λ) / (∏yᵢ!)
            = λ^57 exp(-6λ) / (∏yᵢ!)
```

**(b)**

``` r
counts <- c(8, 12, 7, 10, 9, 11)
n <- length(counts)
sum_counts <- sum(counts)

# Empirical MLE
lambda_values <- seq(1, 20, by = 0.1)
log_lik <- sum_counts * log(lambda_values) - n * lambda_values

lambda_mle <- lambda_values[which.max(log_lik)]
lambda_mle
# Analytical answer: mean(counts) = 9.5
```

### Answer 20

**(a)** **Prefer empirical approach when:** - Analytical derivative is
difficult or impossible to solve - Quick exploratory analysis is
needed - Teaching/learning purposes (more intuitive) - Constrained
parameter space (e.g., λ must be positive)

**(b)** **Prefer analytical approach when:** - Derivative can be solved
algebraically - Need exact solution (not approximation) - Working with
standard distributions with known MLEs - Need theoretical properties
(standard errors, etc.)

**(c)** **Advantages:** - **Empirical:** Simple to implement, works for
any distribution, no calculus needed, easy to visualize -
**Analytical:** Exact answer, faster computation, provides closed-form
expressions, enables theoretical analysis

------------------------------------------------------------------------

## Part D: Simulation and Random Sampling

### Answer 21

**(a) & (c)**

``` r
set.seed(42)
rolls <- sample(1:6, size = 1000, replace = TRUE)
```

**(b)**

``` r
# Using table
pmf_table <- table(rolls) / 1000

# Using janitor
library(janitor)
pmf_janitor <- tabyl(rolls)

# Display
pmf_table
```

### Answer 22

**(a) & (b)**

``` r
set.seed(123)

sample_sizes <- c(10, 100, 1000, 10000)
sample_means <- numeric(length(sample_sizes))

for (i in seq_along(sample_sizes)) {
  n <- sample_sizes[i]
  sample_data <- rnorm(n, mean = 50, sd = 10)  # sd = sqrt(100)
  sample_means[i] <- mean(sample_data)
}

# Create comparison
results <- data.frame(
  sample_size = sample_sizes,
  sample_mean = sample_means
)
print(results)
```

**(c)** The sample mean should approach **μ = 50** as n increases (Law
of Large Numbers).

### Answer 23

**(a)**

``` r
set.seed(551)
weibull_sample <- rweibull(10000, shape = 2, scale = 5)
```

**(b)**

``` r
# Empirical quantities
empirical_mean <- mean(weibull_sample)
empirical_var <- var(weibull_sample)

cat("Empirical Mean:", empirical_mean, "\n")
cat("Empirical Variance:", empirical_var, "\n")
```

**(c)**

``` r
# Empirical 75th percentile
empirical_q75 <- quantile(weibull_sample, 0.75)

# Theoretical 75th percentile
theoretical_q75 <- qweibull(0.75, shape = 2, scale = 5)

# Compare
cat("Empirical Q75:", empirical_q75, "\n")
cat("Theoretical Q75:", theoretical_q75, "\n")
cat("Difference:", abs(empirical_q75 - theoretical_q75), "\n")
```

### Answer 24

**(a)**

``` r
set.seed(42)
n_sim <- 5000

# Flip coins
coins <- sample(c("H", "T"), size = n_sim, replace = TRUE)

# Generate values based on coin flips
results <- numeric(n_sim)
for (i in 1:n_sim) {
  if (coins[i] == "H") {
    results[i] <- rnorm(1, mean = 10, sd = 2)  # sd = sqrt(4)
  } else {
    results[i] <- rnorm(1, mean = 20, sd = 3)  # sd = sqrt(9)
  }
}

# More efficient vectorized version:
heads_idx <- coins == "H"
results[heads_idx] <- rnorm(sum(heads_idx), mean = 10, sd = 2)
results[!heads_idx] <- rnorm(sum(!heads_idx), mean = 20, sd = 3)
```

**(b)**

``` r
overall_mean <- mean(results)
overall_mean  # Should be approximately 15 (average of 10 and 20)
```

### Answer 25

``` r
set.seed(100)

# (a) Generate number of ships
num_ships <- rpois(1, lambda = 4)

# (b) Generate unloading times and sum
if (num_ships > 0) {
  unload_times <- runif(num_ships, min = 2, max = 6)
  total_time <- sum(unload_times)
} else {
  total_time <- 0
}

cat("Number of ships:", num_ships, "\n")
cat("Total unloading time:", total_time, "hours\n")
```

------------------------------------------------------------------------

## Part E: Integrated Problems

### Answer 26

**(a)** **Why Log-Normal?** - Purchase amounts are strictly positive
(cannot be negative) - Often right-skewed (many small purchases, few
large ones) - Log-Normal naturally models multiplicative processes -
Normal distribution could give negative values, which is unrealistic for
prices - Log-Normal is bounded below by zero, which matches real-world
constraints

**(b)**

``` r
purchases <- c(25, 45, 30, 120, 55, 80, 35, 200, 60, 90)

# Transform to log scale for MLE
log_purchases <- log(purchases)

# MLE estimates (for log-normal, we estimate parameters of the normal for log(X))
mu_hat <- mean(log_purchases)
sigma2_hat <- var(log_purchases) * (length(log_purchases) - 1) / length(log_purchases)
# Note: var() uses n-1, but MLE uses n
sigma_hat <- sqrt(sigma2_hat)

cat("MLE mu:", mu_hat, "\n")
cat("MLE sigma:", sigma_hat, "\n")

# Generate 1000 samples from fitted distribution
set.seed(42)
fitted_samples <- rlnorm(1000, meanlog = mu_hat, sdlog = sigma_hat)

# Visualize
hist(fitted_samples, breaks = 30, main = "Fitted Log-Normal Distribution",
     xlab = "Purchase Amount")
```

### Answer 27

``` r
set.seed(12345)

# Parameters
n_simulations <- 1000
sample_size <- 30
beta_true <- 5

# Storage for sample means
sample_means <- numeric(n_simulations)

# Simulation loop
for (i in 1:n_simulations) {
  # Generate sample
  sample_data <- rexp(sample_size, rate = 1/beta_true)
  
  # Calculate and store sample mean
  sample_means[i] <- mean(sample_data)
}

# Calculate mean of sample means
mean_of_means <- mean(sample_means)

# Compare to true mean
cat("True population mean (beta):", beta_true, "\n")
cat("Mean of 1000 sample means:", mean_of_means, "\n")
cat("Difference:", abs(mean_of_means - beta_true), "\n")

# Visualize
hist(sample_means, breaks = 30, 
     main = "Distribution of Sample Means",
     xlab = "Sample Mean",
     col = "lightblue")
abline(v = beta_true, col = "red", lwd = 2, lty = 2)
abline(v = mean_of_means, col = "blue", lwd = 2)
legend("topright", 
       legend = c("True Mean", "Mean of Sample Means"),
       col = c("red", "blue"), lwd = 2, lty = c(2, 1))

# The mean of sample means should be very close to beta_true = 5,
# demonstrating that the sample mean is an unbiased estimator
```
