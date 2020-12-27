from .BluetoothManager import BluetoothManager

class CarController():
	def __init__(self):
		self.shouldPrint = True
		self.bluetoothManager = BluetoothManager()

	def driveAuto(self,prediction):
		if prediction == 0:
			self.forward()
		elif prediction == 1:
			self.forwardRight()
		elif prediction == 2:
			self.forwardLeft()

	def drive(self,function):
		if function != None:
			function = int(function)
		if function==0:
			self.stop()
		elif function==1:
			self.forward()
		elif function==2:
			self.backward()
		elif function==3:
			self.right()
		elif function==4:
			self.left()
		elif function==5:
			self.forwardRight()
		elif function==6:
			self.forwardLeft()
		elif function==7:
			self.backwardRight()
		elif function==8:
			self.backwardLeft()

	def stop(self):
		if self.shouldPrint :
			print("Stop")
		self.bluetoothManager.send(0)

	def forward(self):
		if self.shouldPrint :
			print("Forward")
		self.bluetoothManager.send(1)

	def backward(self):
		if self.shouldPrint :
			print("Backward")
		self.bluetoothManager.send(2)

	def right(self):
		if self.shouldPrint :
			print("Right")
		self.bluetoothManager.send(3)

	def left(self):
		if self.shouldPrint :
			print("Left")
		self.bluetoothManager.send(4)

	def forwardRight(self):
		if self.shouldPrint :
			print("Forward Right")
		self.bluetoothManager.send(5)

	def forwardLeft(self):
		if self.shouldPrint :
			print("Forward Left")
		self.bluetoothManager.send(6)

	def backwardRight(self):
		if self.shouldPrint :
			print("Backward Right")
		self.bluetoothManager.send(7)

	def backwardLeft(self):
		if self.shouldPrint :
			print("Backward Left")
		self.bluetoothManager.send(8)

	def updateSpeed(self,speed):
		if self.shouldPrint :
			print("Update Speed")
		self.bluetoothManager.send(speed)
		
	def close(self):
		self.bluetoothManager.close()