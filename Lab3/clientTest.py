import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# This IP address is the address of the server (i.e. Mac in this case)
client.connect(('172.20.10.2', 8080))
client.send('I am CLIENT\n'.encode())
from_server = client.recv(4096)
client.close()
print(from_server.decode())
