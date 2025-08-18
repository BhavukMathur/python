# Input (do not edit)
import numpy as np  # For mathematical operations
x, y = [int(value.strip()) for value in input().split(',')]

# Define function to classify the property
def classify_property(x, y):

    # Assuming uniform distribution from 1 to 10
    a, b = 1, 10
    values = np.arange(a, b + 1)
    mean = (a + b) / 2
    std_dev = np.std(values)

    high_threshold = mean + std_dev
    low_threshold = mean - std_dev

    if x > high_threshold and y > high_threshold:
        return "Posh Property"
    elif x < low_threshold and y < low_threshold:
        return "Low-End Property"
    else:
        return "Standard Property"

# Print output (do not edit)
print(classify_property(x, y))