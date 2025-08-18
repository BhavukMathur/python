# Taking Input 
f = float(input())

class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    @classmethod
    def from_fahrenheit(cls, f):
        celsius = round(((f - 32) * 5) / 9, 2)
        return cls(celsius)

val = Temperature.from_fahrenheit(f)

print(val.celsius)
