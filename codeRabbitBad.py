# bad_code.py

import os, sys   # bad practice: multiple imports on same line, unused imports

def add_numbers(a, b):
    return a+b    # no spaces around operator, missing type hints, missing docstring


def divide_numbers(a, b):
    return a / b   # no error handling (ZeroDivisionError)


def unused_function(x, y, z):
    result = x + y
    # z is never used
    return result


class calculator:   # class name should be PascalCase
    def __init__(self):
        self.value = 0
    
    def add(self, number):
        self.value=self.value+number   # spacing & style issues
        return self.value
    
    def Subtract(self, number):   # inconsistent naming convention
        self.value -= number
        return self.value

print(divide_numbers(4, 2))