import socket
import time

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        #s.sendall(b'Hello, world\n')
        #time.sleep(1)
        data = s.recv(1024)

        print('Received', float(data.decode('utf_8')))
        #print(type(float(data.decode('utf_8'))))