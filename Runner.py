from builtins import KeyboardInterrupt

from Classes.Car import Car

import sys

from multiprocessing import Process

if __name__ == "__main__":

	if len(sys.argv) > 1 :
		if sys.argv[1] in ['m','a']:
			car = Car(mode=sys.argv[1],showStream=True)
		else: 
			print("Invalid Driving Mode")
			print("a for Autonomous Mode")
			print("m for Manual Mode")
			sys.exit()
	else:
		car = Car()
	try:
		p = Process(target=car.start())
		p.start()
		p.join()
	
		
	except KeyboardInterrupt:
		print("\nKeyboard Interrupt")
	finally:
		print("\nReleasing Resources")
		car.close()
		print("Exiting ...")
#sudo netstat -nlp | grep 8000
#sudo kill -9 10869