import math, ast
data = ast.literal_eval(input())

# Input has been taken for you as a tuple

# Write your code below

#input as x, o, and u

x, o, u = data

if (o <= 0):
    print ("Invalid input")
else:
    z = (x - u) / o
    print (round (z, 2))