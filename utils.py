import socket

def get_computer():
    return socket.gethostname()

def is_galois():
    return get_computer() == "Galois"
