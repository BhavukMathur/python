#Input: [[3, 4], [1, 0], [0, 1]]
import math, ast
data = ast.literal_eval(input())
v, b1, b2 = data

def change_of_basis(v, b1, b2):
    # Extract components
    a, c = b1          # first column
    b, d = b2          # second column
    det = a * d - b * c

    if det == 0:
        return "Invalid input"

    # Inverse of 2×2 matrix [a b; c d] is (1/det)[d -b; -c a]
    inv = [[ d / det, -b / det],
           [-c / det,  a / det]]

    # Multiply inverse by v
    c1 = inv[0][0] * v[0] + inv[0][1] * v[1]
    c2 = inv[1][0] * v[0] + inv[1][1] * v[1]
    return [round(c1, 2), round(c2, 2)]


print(change_of_basis(v, b1, b2))