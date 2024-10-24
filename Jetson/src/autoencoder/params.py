import os 

PLANT_SERVER_IP = os.getenv("PLANT_SERVER_IP", "opc.tcp://localhost:4848")
TS_PLANT = int(os.getenv("TS_PLANT", 1))