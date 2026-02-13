import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms
# -----------------------------------

def normal_histogram(n):
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.title("Normal(0,1) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    return data


def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.title("Uniform(0,10) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    return data


def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.title("Bernoulli(0.5) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    return np.mean(data)


def sample_variance(data):
    return np.var(data, ddof=1)  # n-1 denominator


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    data = np.sort(data)
    n = len(data)

    minimum = data[0]
    maximum = data[-1]
    median = np.median(data)

    # Consistent quartile definition to match test:
    # percentile method that gives Q1=2, Q3=4 for [1,2,3,4,5]
    q1 = np.percentile(data, 25, method="nearest")
    q3 = np.percentile(data, 75, method="nearest")

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    return np.cov(x, y, ddof=1)[0, 1]


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    return np.cov(x, y, ddof=1)
