import math, ast
data = ast.literal_eval(input())
x, y = data


def gradient(x, y):
    dfdx = 2 * x + 2 * y
    dfdy = 2 * y + 2 * x
    return [round(dfdx, 2), round(dfdy, 2)]


print(gradient(x, y))