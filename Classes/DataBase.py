import datetime
from .config import storageConfig,output
import cv2
class DataBase:

	def __init__(self):
		self.file = open(storageConfig['filename'],'w+')
		self.file.write('X,y,endcodedY\n')
		self.imageSaved = 0

	def show(self,path):
		img = cv2.imread(path)
		cv2.imshow('image',img)
		cv2.waitKey(0)

	def save(self,X,y,encoded):
		if y == None or encoded not in [output['forward'],output['forwardRight'],output['forwardLeft']] :
			return
		pth = storageConfig['dataDir']+'/'+str(datetime.datetime.now())+'.jpg'
		X.save(pth,"JPEG",quality=80)
		self.file.write('{},{},{}\n'.format(pth,y,encoded))
		self.imageSaved=+1
		#self.show(pth)

	def close(self):
		self.file.close()