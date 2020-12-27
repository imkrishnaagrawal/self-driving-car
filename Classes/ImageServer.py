from PIL import Image
import struct
import serial
import datetime
import socket
import io
import pexpect
from .config import serverConfig,clientConfig

class ImageServer:

	def __init__(self):
		self.server_socket = socket.socket()
		self.server_socket.bind((serverConfig['host'], serverConfig['port']))
		self.server_socket.listen(0)
		print("Wating For Client ...")
		#subprocess.Popen(['ssh', 'pi@raspberrypi.local', 'python refactored.py'],
        #                stdin=subprocess.PIPE)
	
		child = pexpect.spawn('ssh '+clientConfig['user']+'@'+clientConfig['host']+' "python '+clientConfig['remoteFilename']+' '+serverConfig['host']+' '+str(serverConfig['port'])+'"')
		child.expect(['password: ']) 
		child.sendline (clientConfig['password'])
		self.connection = self.server_socket.accept()[0].makefile('rb')
		print("Connected To ImageClient ")

	def getImage(self):

		# Read the length of the image as a 32-bit unsigned int. If the
		# length is zero, quit the loop
		image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
		if not image_len:
			return
		# Construct a stream to hold the image data and read the image
		# data from the self.self.connection
		image_stream = io.BytesIO()
		image_stream.write(self.connection.read(image_len))
		# Rewind the stream, open it as an image with PIL and do some
		# processing on it
		image_stream.seek(0)
		image = Image.open(image_stream).convert('RGB')
		return image
		#pth = './img/'+str(datetime.datetime.now())+'.jpg'
		#image.save(pth,"JPEG",quality=80)
		#return pth


	def close(self):
		self.connection.close()
		self.server_socket.close()