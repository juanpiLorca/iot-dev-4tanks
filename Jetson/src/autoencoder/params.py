import os 

PLANT_SERVER_IP = os.getenv("PLANT_SERVER_IP", "opc.tcp://192.168.0.122:4848")
TS_PLANT = int(os.getenv("TS_PLANT", 1))
USE_AUTOENCODER = bool(os.getenv("USE_AUTOENCODER", True))
ADD_NOISE = bool(os.getenv("ADD_NOISED", True))