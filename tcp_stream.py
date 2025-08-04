import socket
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import sys
import time

class TcpStream:
    def __init__(self, host='127.0.0.1', port=1756):
        self.HOST = host
        self.PORT = port

    def listenToStream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                sock.connect((self.HOST, self.PORT))
                break
            except ConnectionRefusedError:
                time.sleep(2)
        try:
            while True:
                length_bytes = sock.recv(4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, byteorder='little')
                data = b''
                while len(data) < length:
                    packet = sock.recv(length - len(data))
                    if not packet:
                        break
                    data += packet
                try:
                    image = Image.open(BytesIO(data))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    yield frame
                except Exception as e:
                    continue
        except Exception as e:
            sock.close()
            sys.exit(1)
        finally:
            sock.close()
