import socket
# generate random floating point values
from random import seed
from random import random
# seed random number generator
import time
seed(1)

def rng():
    value = random()
    return value

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            #data = conn.recv(1024)
            #if not data:
            #    break
            data = str(rng()).encode('utf_8')
            conn.sendall(data)
            time.sleep(0.1)
            #print(data)