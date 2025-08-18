print("Hello world")

x = 2;
y = 3;
name = "user1"

# Exponent
print (x ** y)

# String valueOf
print (name + str(x))

# List
someList = ['apple', 'banana', 'cheeku', 'dates']
someList.append('papaya')

for item in someList:
    print (item)

print(someList[1:3])    #starting index < 3 index
print(someList[-1])     #starting from the end

# HashMap
someHashMap = {
    'apple': 'red',
    'banana': 'yellow',
    'cheeku': 'brown',
    'dates': 'brown'
}

for key,value in someHashMap.items():
    print(key + " : " + value)

print(someHashMap.keys())

for i in range(5):
    print("Done.." + str(i))
