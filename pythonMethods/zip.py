# Sample Input:
# [1, 2, None]
# ['x', 'y', 'z']


# Sample Output:
# [(1, 'x'), (2, 'y')]

from ast import literal_eval

# Taking Inputs 
a = literal_eval(input())
b = literal_eval(input())

def filtered_zip(a, b):
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    # Write your code here
    

# Print the output
print(filtered_zip(a, b))