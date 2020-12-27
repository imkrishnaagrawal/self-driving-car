from .ImageServer import ImageServer
from .KeyBoradInput import KeyBoradInput
from .DataBase import DataBase
from .CarController import CarController
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import sys
import time
from multiprocessing.pool import ThreadPool 
import math

class HaarCascadeClassifier:
	def __init__(self,xml,height,width):
		self.classifier = cv2.CascadeClassifier(xml)
		self.height = height
		self.width = width
		#self.ay = 236.28183027255858
		#self.v0 = 143.97335006860723
		self.v0 = 119.865631204
		self.ay = 332.262498472 
		self.alpha = 8.0* math.pi / 180
		self.distance = 1
	
	def updateDistance(self,v,image):
		self.distance =  (self.height / math.tan(self.alpha + math.atan((v - self.v0) / self.ay)))
		print(self.distance)
		if self.distance > 0:
			cv2.putText(image, "%.1fcm" % self.distance,
                        (image.shape[1] - self.width, image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		 

	def detectObject(self,X):
		image = np.array(X)
		image = image[:, :, ::-1].copy() 
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		v = 0
		threshold = 80
		cascade = self.classifier.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30))
		for (x_pos, y_pos, width, height) in cascade:
			cv2.rectangle(image, (x_pos + 5, y_pos + 5), (x_pos + width - 5, y_pos + height - 5), (255, 255, 255), 2)
			v = y_pos + height - 5
			if width / height == 1:
				cv2.putText(image, 'STOP', (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				roi = gray[y_pos + 10:y_pos + height - 10, x_pos + 10:x_pos + width - 10]
				mask = cv2.GaussianBlur(roi, (25, 25), 0)
				(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
				if maxVal - minVal > threshold:
					cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)
					if 1.0 / 8 * (height - 30) < maxLoc[1] < 4.0 / 8 * (height - 30):
						cv2.putText(image, 'Red', (x_pos + 5, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
						self.red = True
					# Green light
					elif 5.5 / 8 * (height - 30) < maxLoc[1] < height - 30:
						cv2.putText(image, 'Green', (x_pos + 5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
						self.green = True
					# yellow light
					elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
					   cv2.putText(image, 'Yellow', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
					   self.yello = True
		if v > 0 :
			self.updateDistance(v,image)
		cv2.imshow('img',image)
		cv2.waitKey(1)
		return self.distance

	

class Car:

	def __init__(self,mode='m',showStream=True):
		self.server = ImageServer()
		self.keyBoradInput = KeyBoradInput()
		self.database = DataBase()
		self.carController = CarController()
		self.mode = mode
		self.isRunning = False
		self.showStream = showStream
		self.pool = ThreadPool(processes=2)
		self.stop_sign=HaarCascadeClassifier('stop_sign.xml',10.5,300)
		self.traffic_sign=HaarCascadeClassifier('cascade.xml',4,700)	
	
		if mode == 'm':
			print('Ready to drive in Manual mode \nPress Keys To Drive')
		elif mode == 'a':
			print('Ready to drive in Autonomous Mode')

	def start(self):
		self.isRunning = True
		if self.mode == 'm':
			self.collectData()
		elif self.mode == 'a':
			self.driveAuto()

	def stop(self):
		self.isRunning = False

	def read_image(self,path):
		return mpimg.imread(path)

	def show_objects(self,bbox, label, conf,image):	
		print(bbox, label, conf)
		out = draw_bbox(image, bbox, label, conf)
		cv2.imshow("object_detection", out)
		cv2.waitKey(10)

	def collectData(self):
		while self.isRunning:
			y,encoded = self.keyBoradInput.getCommand()
			if y == 'q':
				return
			self.carController.drive(y)
			X = self.server.getImage()
			self.database.save(X,y,encoded)
			stopDistance    = self.stop_sign.detectObject(X)
			trafficDistance = self.traffic_sign.detectObject(X)
			#print('Stop and Traffic ',stopDistance,trafficDistance)


	def driveAuto(self):
		model = tf.keras.models.load_model('donket3.h5')
		print(model)
		while self.isRunning:
			X = self.server.getImage()
			
			stopDistance = self.stop_sign.detectObject(X)
			X = np.array(X)
			print('Shape and Type',X.shape,type(X))
			#X = self.maskColor(X)
			X = np.array([X])
			X = cv2.normalize(X, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			X = X[0].reshape((-1,240,320,3))
			y = np.argmax(model.predict(X))
			print('-------------Prediction : ',y,'-------------')
			self.carController.driveAuto(y)
			
	
	def maskColor(self,img):
		img = cv2.bilateralFilter(img,9,75,75)
		lower = np.array([0,0,0])  #-- Lower range --
		upper = np.array([90,90,90])  #-- Upper range --
		mask = cv2.inRange(img, lower, upper)
		res = cv2.bitwise_and(img, img, mask= mask)  #-- Contains pixels having the gray color--
		return self.getROI(res)
		
	def getROI(self,image):
		#return image
		#image=cv2.GaussianBlur(image,(5,5),0)
		image=cv2.bilateralFilter(image,9,75,75)
		polygons=np.array([ [(0,240),(0,150),(70,100),(250,100),(320,150),(320,240)]]) #[(0,240),(150,100),(300,240)]  ])
		mask=np.zeros_like(image)
		cv2.fillPoly(mask,polygons,(255, 255, 255))
		return cv2.bitwise_and(mask,image)

	def close(self):
		self.server.close()
		self.keyBoradInput.close()
		self.database.close()
		self.carController.close()
		cv2.destroyAllWindows()