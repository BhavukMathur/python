# Taking Inputs
name = input()
balance = float(input())

class BankAccount:
    # Write your code here
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance

    def show_balance(self):
        print(f"Name: {self.name}, Balance: {self.balance}")


# Create object and print result
account = BankAccount(name, balance)
account.show_balance()
