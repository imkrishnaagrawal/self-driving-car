import bluetooth
from .config import bluetoothConfig

class BluetoothManager:

	def __init__(self):
		
		if bluetoothConfig['bd_addr'] :
			bd_addr = bluetoothConfig['bd_addr']
		else:
			print("Searching for Bluetooth Device...")
			selected = None
			nearby_devices = bluetooth.discover_devices()
			for i , device in enumerate(nearby_devices):
				print(i," => ",bluetooth.lookup_name(device))
				if bluetoothConfig['name'] != None and bluetoothConfig['name'] in bluetooth.lookup_name(device):
					selected = i
			
			if bluetoothConfig['name'] == None:
				selected = int(input("> "))
			print("Connecting to ", bluetooth.lookup_name(nearby_devices[selected]))
			bd_addr = nearby_devices[selected]
			
		self.serialconn = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
		self.serialconn.connect((bd_addr,bluetoothConfig['port']))
		print("Connected To Bluetooth")

	def send(self,data):
		self.serialconn.send(chr(data).encode()) #write(chr(data).encode())

	def close(self):
		self.serialconn.close()