import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

df = pd.read_csv('data/dataset.csv', names=('x', 'y'))
slope, intercept, r_value, p_value, std_err = stats.linregress(df)

learning_rate = float(0.0001)
initial_theta_0 = float(0)
initial_theta_1 = float(0)
nbr_iterations = 2000
M = df.x.size


def predict(x_item):
    return slope * x_item + intercept


fitLine = predict(df.x)


def calculate_cost_function(theta_0, theta_1):
    global_cost = 0

    for i in df.x.index:
        global_cost += ((theta_0 + theta_1 * df.x[i]) - df.y[i]) ** 2

    return (1 / 2 * M) * global_cost


def calculate_partial_derivatives(old_theta_0, old_theta_1):
    der_theta_0 = float(0)
    der_theta_1 = float(0)

    for i in df.x.index:
        der_theta_0 += old_theta_0 + (old_theta_1 * df.x[i]) - df.y[i]
        der_theta_1 += (old_theta_0 + (old_theta_1 * df.x[i]) - df.y[i]) * df.x[i]

    der_theta_0 *= (1 / M)
    der_theta_1 *= (1 / M)

    return der_theta_0, der_theta_1


axes = plt.axes()
axes.grid()

plt.scatter(df.x, df.y)
plt.plot(df.x, fitLine, c='r')
plt.show()
