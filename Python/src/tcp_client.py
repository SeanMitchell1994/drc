import socket
import time
import matplotlib.pyplot as plt

def Process_Data(stream):
    print('Received', float(stream.decode('utf_8')))
    data = float(stream.decode('utf_8'))
    #data = 1 + data
    #print('Processed ', data)

def main():

    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            #s.sendall(b'Hello, world\n')
            #time.sleep(1)
            data = s.recv(1024)
            Process_Data(data)
        
        
if __name__ == "__main__":
    main()