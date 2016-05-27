from statistics import mean
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# graph style
style.use('fivethirtyeight')

def create_dataset(size, variance, step=2, correlation='positive'):
	'''

	Args:
		size (int): the number of items in the dataset
		variance (float): the range of values in the dataset
		step (float): the increment of the values
		correlation (Optional[str]): the slope of the line (positive or negative)
	'''
	val = 1
	ys = []
	for i in range(size):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'positive':
			val += step
		elif correlation and correlation == 'negative':
			val -= step
	xs = [i for i in range(size)]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_line(xs, ys):
	'''Calculates the slope of the best-fit line.
	'''
	numerator = mean(xs) * mean(ys) - mean(xs * ys)
	denominator = mean(xs)**2 - mean(xs**2)
	m = numerator / denominator
	b = mean(ys) - m * mean(xs)
	print('Slope: {}, Y-Intercept: {}'.format(m, b))
	return m, b

def squared_error(ys_original, ys_line):
	'''Calculates the square error of a data set.

	Args:
		ys_original (array): the output values from the training set
		ys_line (array): the output values from a linear regression model
	'''
	return sum((ys_line - ys_original)**2)

def coef_det(ys_original, ys_line):
	'''Calculates the coefficient of determination, which is one measure of
	how well a model performs (relative to the mean).

	Args:
		ys_original (array): the output values from the training set
		ys_line (array): the output values from a linear regression model
	'''
	y_mean_line = [mean(ys_original) for y in ys_original]			# basically, a horizontal line at the mean value
	squared_error_regression = squared_error(ys_original, ys_line)
	squared_error_y_mean = squared_error(ys_original, y_mean_line)
	return 1 - squared_error_regression / squared_error_y_mean

xs, ys = create_dataset(40, 40, 2, correlation='positive')
print(xs, ys)
m, b = best_fit_line(xs, ys)

regression_line = [m * x + b for x in xs]

cd = coef_det(ys, regression_line)
print('Coefficient of determination:', cd)

# plot the original data and the best-fit line
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
