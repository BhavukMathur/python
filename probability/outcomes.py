import ast

# Input has been taken
data = ast.literal_eval(input())

# The outcomes and probabilities have been extracted from the input
outcomes, probabilities = data[0], data[1]

# Write your code below

if(abs(sum(probabilities)) > 1):
    print("Invalid input")
else:
    result = sum(x * p for x,p in zip(outcomes, probabilities))
    print(round(result, 2))    