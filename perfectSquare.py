import math

def is_perfect_square(num):
    sqrt = (int)(num ** 0.5)
    num2 = sqrt * sqrt
    return (num2 == num)

n = 37
print(is_perfect_square(n))