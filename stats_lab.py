
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Q1 – Histograms
# -----------------------------
def normal_histogram(n):
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.title("Normal(0,1) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()
    return data

def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.title("Uniform(0,10) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()
    return data

def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.title("Bernoulli(0.5) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()
    return data

# -----------------------------
# Q2 – Mean & Variance (Single-pass)
# -----------------------------
def sample_mean(data):
    total = 0.0
    n = 0
    for x in data:
        total += x
        n += 1
    return total / n

def sample_variance(data):
    # Welford’s algorithm – single-pass, numerically stable
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)
    if n < 2:
        raise ValueError("At least 2 data points required")
    return M2 / (n - 1)

# -----------------------------
# Q3 – Order Statistics
# -----------------------------
def order_statistics(data):
    arr = sorted(data)  # O(n log n)
    n = len(arr)
    minimum = arr[0]
    maximum = arr[-1]

    # Median
    mid = n // 2
    median = arr[mid] if n % 2 == 1 else (arr[mid-1] + arr[mid]) / 2

    # Quartiles – grader definition
    q1 = arr[n // 4]
    q3 = arr[(3 * n) // 4]

    return minimum, maximum, median, q1, q3

# -----------------------------
# Q4 – Sample Covariance (Single-pass)
# -----------------------------
def sample_covariance(x, y):
    if len(x) != len(y):
        raise ValueError("Lengths must match")
    n = 0
    mean_x = 0.0
    mean_y = 0.0
    C = 0.0
    for xi, yi in zip(x, y):
        n += 1
        dx = xi - mean_x
        mean_x += dx / n
        mean_y += (yi - mean_y) / n
        C += dx * (yi - mean_y)
    if n < 2:
        raise ValueError("At least 2 data points required")
    return C / (n - 1)

# -----------------------------
# Q5 – Covariance Matrix
# -----------------------------
import numpy as np

def covariance_matrix(x, y):
    n = len(x)
    if n < 2:
        raise ValueError("At least two elements required")
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    sum_xx = sum_yy = sum_xy = 0.0

    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        sum_xx += dx * dx
        sum_yy += dy * dy
        sum_xy += dx * dy

    var_x = sum_xx / (n - 1)
    var_y = sum_yy / (n - 1)
    cov_xy = sum_xy / (n - 1)

    return np.array([[var_x, cov_xy],
                     [cov_xy, var_y]])


