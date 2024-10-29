import os 

PLANT_SERVER_IP = os.getenv("PLANT_SERVER_IP", "opc.tcp://192.168.0.122:4840")
TS_PLANT = int(os.getenv("TS_PLANT", 1))