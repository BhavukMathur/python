import ast


def func(data):
    a, b, c = data
    # print(a)    #total
    # print(b)    #passed Maths
    # print(c)    #passed both Math and Science

    if(c > b or b > a):
        return 'Invalid input'
    
    if(b == 0):
        return 'Invalid input'
    
    #Conditional probability
    probab = c / b 
    return round(probab, 2)

# Tuple input has been taken for you
data = ast.literal_eval(input())

# Write your code below, 'data' is the tuple

result = func(data)
print(result)