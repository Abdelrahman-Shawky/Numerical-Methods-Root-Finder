import math
import time
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from math import *


def evaluate(expression, value):
    new_expression = expression.replace("^", "**")
    new_expression = new_expression.replace("ln", "log")
    new_expression = new_expression.replace("exp", "e")
    new_expression = new_expression.replace("x", str(value))
    new_expression = new_expression.replace("e", "exp")
    return eval(str(new_expression))


def selectMethod(method_name, expression, a, b, tol=0.00001, max_iterations=50, single_mode=0):
    if method_name == "Bisection":
        return bisection(expression, a, b, tol, max_iterations, single_mode)
    elif method_name == "False-Position":
        return falsePosition(expression, a, b, tol, max_iterations, single_mode)
    elif method_name == "Newton Raphson":
        return newtonRaphson(expression, a, tol, max_iterations, single_mode)
    elif method_name == "Secant":
        return secant(expression, a, b, tol, max_iterations, single_mode)
    elif method_name == "Fixed Point":
        return fixedPoint(expression, a, tol, max_iterations, single_mode)


def getDerivative(expression):
    new_expression = expression.replace('ln', 'log')
    new_expression = new_expression.replace('^', '**')
    new_expression = new_expression.replace('e', 'exp')
    my_symbols = {'x': Symbol('x', real=True)}
    my_func = parse_expr(new_expression, my_symbols)
    derivative = diff(my_func, my_symbols['x'])
    return str(derivative)


# Implementing Fixed Point Method
def fixedPoint(expression, x_initial, tol=0.00001, max_iterations=50, single_mode=0):
    record = []
    table = [[] for _ in range(max_iterations)]
    count = 0
    gx = expression
    derivative_expression = getDerivative(gx)
    convergence_test = evaluate(derivative_expression, x_initial)
    if convergence_test < -1 or convergence_test > 1:
        return "Invalid Method: Does not converge"
    plotFunction(gx)
    start_time = time.time()
    for i in range(max_iterations):
        x_new = evaluate(expression, x_initial)
        record.clear()
        record.append(count)
        record.append(x_initial)
        record.append(x_new)
        relative_error = abs((x_new - x_initial) / x_new)
        record.append(relative_error)

        for x in record:
            table[count].append(x)
        count += 1

        if relative_error < tol:
            break
        x_initial = x_new

        if single_mode == 1:
            fx_new = evaluate(expression, x_new)
            plt.scatter(x_new, fx_new)
            plt.pause(1)

    fxnew = evaluate(expression, x_new)
    plt.text(x_new, fxnew, "  x = " + str(x_new))
    plt.scatter(x_new, fxnew)
    plt.show(block=False)

    table = list(filter(None, table))
    execution_time = time.time() - start_time
    dictionary = dict()
    dictionary['table'] = table
    execution_time = float("{0:.6f}".format(execution_time))
    dictionary['execution_time'] = str(execution_time)
    return dictionary


# Implementing Newton Raphson Method
def newtonRaphson(expression, x_old, tol=0.00001, max_iterations=50, single_mode=0):
    record = []
    table = [[] for _ in range(max_iterations)]
    count = 0
    derivative_expression = getDerivative(expression)
    plotFunction(expression)
    start_time = time.time()
    for i in range(max_iterations):
        fx = evaluate(expression, x_old)
        fderivative = evaluate(derivative_expression, x_old)
        x_new = x_old - (fx / fderivative)
        record.clear()
        record.append(count)
        record.append(x_old)
        record.append(x_new)
        record.append(fx)
        record.append(fderivative)

        relative_error = abs((x_new - x_old) / x_new)
        record.append(relative_error)

        for x in record:
            table[count].append(x)
        count += 1

        if relative_error < tol:
            break
        x_old = x_new

        if single_mode == 1:
            fx_new = evaluate(expression, x_new)
            plt.scatter(x_new, fx_new)
            plt.pause(1)

    fxnew = evaluate(expression, x_new)
    plt.text(x_new, fxnew, "  x = " + str(x_new))
    plt.scatter(x_new, fxnew)
    plt.show(block=False)

    table = list(filter(None, table))
    execution_time = time.time() - start_time
    dictionary = dict()
    dictionary['table'] = table
    execution_time = float("{0:.6f}".format(execution_time))
    dictionary['execution_time'] = str(execution_time)
    return dictionary


# Implementing Secant Method
def secant(expression, xii, xi, tol=0.00001, max_iterations=50, single_mode=0):
    record = []
    table = [[] for _ in range(max_iterations)]
    count = 0
    plotFunction(expression)
    start_time = time.time()
    for i in range(max_iterations):
        fxi = evaluate(expression, xi)
        fxii = evaluate(expression, xii)
        x_new = xi - (fxi * (xii - xi)) / (fxii - fxi)
        record.clear()
        record.append(count)
        record.append(xii)
        record.append(xi)
        record.append(fxii)
        record.append(fxi)
        record.append(x_new)

        relative_error = abs((x_new - xi) / x_new)
        record.append(relative_error)

        for x in record:
            table[count].append(x)
        count += 1

        if relative_error < tol:
            break
        xii = xi
        xi = x_new

        if single_mode == 1:
            fxi_plot = evaluate(expression, xi)
            fxii_plot = evaluate(expression, xii)
            plt.scatter(xi, fxi_plot)
            plt.scatter(xii, fxii_plot)
            plt.pause(1)

    fxnew = evaluate(expression, x_new)
    plt.text(x_new, fxnew, "  x = " + str(x_new))
    plt.scatter(x_new, fxnew)
    plt.show(block=False)

    table = list(filter(None, table))
    execution_time = time.time() - start_time
    dictionary = dict()
    dictionary['table'] = table
    execution_time = float("{0:.6f}".format(execution_time))
    dictionary['execution_time'] = str(execution_time)
    return dictionary


# Implementing Bisection Method
def bisection(expression, a, b, tol=0.00001, max_iterations=50, single_mode=0):
    record = []
    table = [[] for _ in range(max_iterations)]
    fa = evaluate(expression, a)
    fb = evaluate(expression, b)
    if fa * fb > 0:
        print("f(a) and f(b) must have different signs")
        return "InputError: f(a) and f(b) must have different signs"
    plotFunction(expression)
    count = 1
    previous_x = 0
    relative_error = math.inf
    start_time = time.time()
    for _ in range(max_iterations):
        c = (a + b) / 2
        record.clear()
        record.append(count)
        record.append(a)
        record.append(b)
        record.append(c)

        fc = evaluate(expression, c)

        if relative_error < tol:
            break

        if fa * fc > 0:
            a = c
            fa = fc

        if fb * fc > 0:
            b = c
            fb = fc

        record.append(fc)
        if count != 1:
            relative_error = abs((c - previous_x) / c)
            record.append(relative_error)
        else:
            record.append(0)
        previous_x = c
        for x in record:
            table[count - 1].append(x)

        count += 1

        if single_mode == 1:
            plt.scatter(previous_x, fc)
            plt.pause(1)

    froot = evaluate(expression, previous_x)
    plt.text(previous_x, froot, "  x = " + str(previous_x))
    plt.scatter(previous_x, froot)
    plt.show(block=False)

    table = list(filter(None, table))
    execution_time = time.time() - start_time
    dictionary = dict()
    dictionary['table'] = table
    execution_time = float("{0:.6f}".format(execution_time))
    dictionary['execution_time'] = str(execution_time)
    return dictionary


# Implementing False Position Method
def falsePosition(expression, a, b, tol=1e-5, max_iterations=50, single_mode=0):
    record = []
    table = [[] for _ in range(max_iterations)]
    fa = evaluate(expression, a)
    fb = evaluate(expression, b)

    if fa * fb > 0:
        print("f(a) and f(b) must have different signs")
        return "InputError: f(a) and f(b) must have different signs"
    count = 1
    previous_x = 0
    relative_error = math.inf
    plotFunction(expression)
    start_time = time.time()
    for _ in range(max_iterations):

        fa = evaluate(expression, a)
        fb = evaluate(expression, b)
        c = ((a * fb) - (b * fa)) / (fb - fa)
        fc = evaluate(expression, c)
        record.clear()
        record.append(count)
        record.append(a)
        record.append(b)
        record.append(c)
        # Check if the above found point is root
        if fc == 0:
            break
        if relative_error < tol:
            break
        # Decide the side to repeat the steps
        elif fc * fa < 0:
            b = c
        else:
            a = c

        record.append(fc)
        if count != 1:
            relative_error = abs((c - previous_x) / c)
            record.append(relative_error)
        else:
            record.append(0)
        previous_x = c
        for x in record:
            table[count - 1].append(x)

        count += 1
        if single_mode == 1:
            plt.scatter(previous_x, fc)
            plt.pause(1)

    froot = evaluate(expression, previous_x)
    plt.text(previous_x, froot, "  x = " + str(previous_x))
    plt.scatter(previous_x, froot)
    plt.show(block=False)

    table = list(filter(None, table))

    execution_time = time.time() - start_time
    dictionary = dict()
    dictionary['table'] = table
    execution_time = float("{0:.6f}".format(execution_time))
    dictionary['execution_time'] = str(execution_time)
    return dictionary


def plotFunction(expression):
    x_list = np.linspace(-10, 10, num=1000)
    plt.figure(num=0, dpi=120)
    plt.plot(x_list, func(x_list, expression))
    plt.title("f(x)")
    plt.xlabel("x")
    plt.grid()
    plt.ylabel("y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)


def func(x, my_input):
    expression = my_input.replace("^", "**")
    expression = expression.replace("cos", "np.cos")
    expression = expression.replace("sin", "np.sin")
    expression = expression.replace("tan", "np.tan")
    expression = expression.replace("ln", "np.log")
    expression = expression.replace("e", "np.exp")
    return eval(expression)
