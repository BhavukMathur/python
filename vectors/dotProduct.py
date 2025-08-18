import math, ast
data = ast.literal_eval(input())
u, v = data

# Input has been taken for you and the lists u and v have been extracted
# Write your code below

if(len(u) != len(v)):
    print("Invalid input")

else:
    magnitudeX = math.sqrt(sum([x ** 2 for x in u]))
    magnitudeY = math.sqrt(sum([y ** 2 for y in v]))

    dotProduct = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    print((round(magnitudeX, 2), round(magnitudeY, 2), round(dotProduct, 2)))