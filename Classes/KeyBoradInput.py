import keyboard
from .config import keys,output

class KeyBoradInput:

	def getCommand(self):
		if keyboard.is_pressed(keys['UP']) and keyboard.is_pressed(keys['RIGHT']):
			return '5',output['forwardRight']

		elif keyboard.is_pressed(keys['UP']) and keyboard.is_pressed(keys['LEFT']):
			return '6',output['forwardLeft']

		elif keyboard.is_pressed(keys['DOWN']) and keyboard.is_pressed(keys['RIGHT']):
			return '7',output['backwardRight']

		elif keyboard.is_pressed(keys['DOWN']) and keyboard.is_pressed(keys['LEFT']):
			return '8',output['backwardLeft']

		elif keyboard.is_pressed(keys['UP']):
			return '1',output['forward']

		elif keyboard.is_pressed(keys['DOWN']):
			return '2',output['backward']

		elif keyboard.is_pressed(keys['RIGHT']):
			return '3',output['right']

		elif keyboard.is_pressed(keys['LEFT']):
			return '4',output['left']

		elif keyboard.is_pressed(keys['STOP']):
			return '0',output['stop']

		elif keyboard.is_pressed(keys['QUIT']):
			return 'q',output['quit']

		elif keyboard.is_pressed(keys['SPEED']):
			return None,output['updateSpeed']
		else:
			return None,None

	def close(self):
		pass # Follows Convention