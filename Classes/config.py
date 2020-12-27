serverConfig = dict(
    host= "krishna-HP-Pavilion-Notebook.local",
    port= 8000
)

clientConfig = dict(
    host = 'raspberrypi.local',
    user = 'pi',
    password = 'pi',
    remoteFilename = 'refactored.py'
)
bluetoothConfig = dict(
    name = 'HC-05', # Optional
    bd_addr = '00:21:13:01:EC:C1', # Optional
    port = 1
)
storageConfig = dict(
    dataDir = './img',
    filename = 'traning.csv'
)

keys = dict(
    UP = 'w',
    DOWN = 's',
    LEFT = 'a',
    RIGHT = 'd',
    QUIT = 'q',
    STOP = "f",
    SPEED = "l"
)

output = dict(
    forward = '100',
    backward = '[]',
    right = '010',
    left = '001',
    forwardRight = '010',
    forwardLeft = '001',
    backwardLeft = '[]',
    backwardRight = '[]',
    stop = '[]',
    quit = '[]',
    updateSpeed = '[]'
)