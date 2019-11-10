import argparse
import sys
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

HOURS_STUDIED = [ .5, .75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5]
FAIL_PASS = [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]


class LinearRegression():
    def __init__(self, X, Y, alpha_init=.1, convergence=.00000001):
        self.learning_rate = alpha_init
        self.convergence = convergence
        self.iteration_counter = 0
        self.X = X
        self.Y = Y
        self.v1 = Y[0]
        self.v0 = Y[1] - Y[0] / X[1] - X[0]
        self.cost_diff = 1
        self.iteration_no = []
        self.cost_for_iter = []
        self.iteration_counter = 0
        self.last_cost = self.simple_cost_function()

        self.initial_values = {
            'X': self.X,
            'Y': self.Y,
            'v1': self.v1,
            'v0': self.v0,
            'learning_rate': self.learning_rate,
            'convergence': self.convergence
        }

    # predictors
    def simple_predictor(self, x):
        return self.v0 + self.v1*x


    #cost functions
    def simple_cost_function(self):
        return (1 / 2 * len(self.X)) * sum([(self.simple_predictor(x) - y)**2 for x, y in zip(self.X, self.Y)])


    #partial derivatives for gradients
    def simple_cost_derived_with_respect_to_v0(self, x, y):
        return self.simple_predictor(x) - y


    def simple_cost_derived_with_respect_to_v1(self, x, y):
        return x * (self.simple_predictor(x) - y)


    # use the gradient of the cost function to get new values with respect to the learning rate
    def update(self):
        d_dv0 = sum([self.simple_cost_derived_with_respect_to_v0(a, b) for a, b in zip(self.X, self.Y)])
        d_dv1 = sum([self.simple_cost_derived_with_respect_to_v1(a, b) for a, b in zip(self.X, self.Y)])

        # We subtract because the derivatives point in direction of steepest ascent
        self.v0 = self.v0 - (self.learning_rate * (d_dv0 / len(self.X)))
        self.v1 = self.v1 - (self.learning_rate * (d_dv1 / len(self.X)))


    def fit(self):
        print(f'Performing gradient descent for linear regression till cost diff < {self.convergence}')
        while np.sqrt(np.square(self.cost_diff)) > self.convergence:
            self.update()

            #Calculate cost for auditing purposes
            cost = self.simple_cost_function()
            self.cost_diff = cost - self.last_cost

            if self.iteration_counter % 100 == 0:
                print(f"iteration: {self.iteration_counter}")
                print(f"cost: {cost}")
                self.iteration_no.append(self.iteration_counter)
                self.cost_for_iter.append(cost)


            self.last_cost = cost
            self.iteration_counter = self.iteration_counter + 1

        output_values = {
            'v1': self.v1,
            'v0': self.v0,
            'iteration_no': self.iteration_no,
            'cost_for_iter': self.cost_for_iter
        }

        return_vals = {
            'in': self.initial_values,
            'out': output_values
        }

        return return_vals


def create_line(x, v0, v1):
    return [(x, v0+v1*x) for x in x]


def create_simple_linear_regression_plots():
    #solve the simple linear regression
    x = HOURS_STUDIED
    y = FAIL_PASS
    scatter_data = { 'x': x, 'y': y}

    # create best fit using ML
    regression = LinearRegression(x, y)
    simple_linear_reg_info = regression.fit()
    v0, v1, iteration_no, cost_per_iter = (
        simple_linear_reg_info['out']['v0'],
        simple_linear_reg_info['out']['v1'],
        simple_linear_reg_info['out']['iteration_no'],
        simple_linear_reg_info['out']['cost_for_iter']
    )
    x, y = zip(*create_line(scatter_data['x'], v0, v1))
    regression_line_data = {'x': x, 'y': y}

    #plot cost as a measure of iterations
    cost_per_iter_line_data = {'x': iteration_no, 'y': cost_per_iter}

    # actually graph the stuff
    plt.figure()
    plt.plot(scatter_data['x'], scatter_data['y'], '.')
    plt.plot(regression_line_data['x'], regression_line_data['y'], 'b-', label='simple linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.savefig('simple_linear_regression.png')
    plt.figure()
    plt.plot(cost_per_iter_line_data['x'], cost_per_iter_line_data['y'], 'b-', label='Cost for iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (J)')
    plt.legend(loc='upper right')
    plt.savefig('simple_linear_regression_cost_per_iteration.png')


class LogisticRegression:
    def __init__(self, X, T, alpha_init=0.1, convergence=.00000001):
        self.learning_rate = alpha_init
        self.convergence = convergence
        self.X = X
        self.T = T
        print(self.X)
        self.v0 = np.random.rand()
        self.v1 = np.random.rand()
        self.cost_diff = 1
        self.iteration_no = []
        self.cost_for_iter = []
        self.iteration_counter = 0
        self.last_cost = self.cost()

        self.initial_values = {
            'X': self.X,
            'T': self.T,
            'v0': self.v0,
            'v1': self.v1,
            'learning_rate': self.learning_rate,
            'convergence': self.convergence
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predictor(self, x):
        return self.sigmoid(self.v0 + x*self.v1)

    def cost(self):
        return ( 1 / len(self.X) * sum([
                -t * np.log(self.predictor(x)) - (1 - t) * np.log(1 - self.predictor(x))
                for x, t in zip(self.X, self.T)
            ])
        )

    def cost_derived_with_respect_to_v0(self, x, t):
        return self.predictor(x) - t

    def cost_derived_with_respect_to_v1(self, x, t):
        return x * (self.predictor(x) - t)

    def update(self):
        d_dv0 = sum([self.cost_derived_with_respect_to_v0(a, b) for a, b in zip(self.X, self.T)])
        d_dv1 = sum([self.cost_derived_with_respect_to_v1(a, b) for a, b in zip(self.X, self.T)])

        # We subtract because the derivatives point in direction of steepest ascent
        self.v0 = self.v0 - (self.learning_rate * (d_dv0 / len(self.X)))
        self.v1 = self.v1 - (self.learning_rate * (d_dv1 / len(self.X)))

    def fit(self):
        print(f'Performing gradient descent for logistic regression till cost diff < {self.convergence}')
        while np.sqrt(np.square(self.cost_diff)) > self.convergence:
            self.update()

            cost = self.cost()
            self.cost_diff = cost - self.last_cost

            if self.iteration_counter % 100 == 0:
                print(f'iteration: {self.iteration_counter}')
                print(f'cost: {cost}')
                self.iteration_no.append(self.iteration_counter)
                self.cost_for_iter.append(cost)

            self.last_cost = cost
            self.iteration_counter = self.iteration_counter + 1

        output_values = {
            'v0': self.v0,
            'v1': self.v1,
            'iteration_no': self.iteration_no,
            'cost_for_iter': self.cost_for_iter
        }

        return_vals = {
            'in': self.initial_values,
            'out': output_values
        }

        return return_vals


def create_logistic_curve(x, v0, v1):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_y_for(x):
        return sigmoid(v0 + x*v1)

    return [(x, get_y_for(x)) for x in x]


def create_logistic_regression_plots():
    #solve the simple linear regression
    x = HOURS_STUDIED
    y = FAIL_PASS
    scatter_data = { 'x': x, 'y': y}

    # create best fit using ML
    regression = LogisticRegression(np.array(x), np.array(y))
    logistic_regression_info = regression.fit()
    v0, v1, iteration_no, cost_per_iter = (
        logistic_regression_info['out']['v0'],
        logistic_regression_info['out']['v1'],
        logistic_regression_info['out']['iteration_no'],
        logistic_regression_info['out']['cost_for_iter']
    )
    x, y = zip(*create_logistic_curve(scatter_data['x'], v0, v1))
    regression_line_data = {'x': x, 'y': y}

    #plot cost as a measure of iterations
    cost_per_iter_line_data = {'x': iteration_no, 'y': cost_per_iter}

    # actually graph the stuff
    plt.figure()
    plt.plot(scatter_data['x'], scatter_data['y'], '.')
    plt.plot(regression_line_data['x'], regression_line_data['y'], 'b-', label='logistic regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.savefig('logistic_regression.png')
    plt.figure()
    plt.plot(cost_per_iter_line_data['x'], cost_per_iter_line_data['y'], 'b-', label='Cost for iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (J)')
    plt.legend(loc='upper right')
    plt.savefig('logistic_regression_cost_per_iteration.png')


def create_plots():
    create_simple_linear_regression_plots()
    create_logistic_regression_plots()


create_plots()
