import math, ast


def binProbab(data):
    n, k, p = data

    if not (isinstance(n, int) and n >= 0):
        print ("Invalid input")
        return
    if not (isinstance(k, int) and 0 <= k <= n):
        print ("Invalid input")
        return
    if not (isinstance(p, float) or isinstance(p, int)) or not (0 <= p <= 1):
        print ("Invalid input")
        return


    # Calculate combinations C(n, k)
    combinations = math.comb(n, k)
    

    # Calculate probability of k successes
    probability_of_successes = p ** k
    
    # Calculate probability of n-k failures
    probability_of_failures = (1 - p) ** (n - k)
    
    # Calculate binomial probability
    result = combinations * probability_of_successes * probability_of_failures
    
    # Round to 2 decimal places
    result = round(result, 2)
    print (result)
    
    return result

# The tuple has been taken as input for you
# Write your code below

data = ast.literal_eval(input())
binProbab(data)
