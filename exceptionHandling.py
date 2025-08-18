# Taking inputs
a = int(input())
b = int(input())

def safe_divide(a, b):
    try:
        print('')
        result = a/b
        return float(result)
    except:
        return "Error: division by zero"
    finally:
        print("Operation attempted")
    # Write your code here



# Print the output
print(safe_divide(a, b))