def fibonacci(n):
    a = 0
    b = 1
    print(a)
    print(b)

    for val in range(n):
        c = a + b
        print(c)
        a = b
        b = c

n = 10
fibonacci(n)