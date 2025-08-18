import socket

def get_hostname():
    return socket.gethostname()

# Example usage:
if __name__ == "__main__":
    print("Hostname:", get_hostname())
