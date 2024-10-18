import os 

plant_server_ip = os.getenv("PLANT_SERVER_IP", "opc.tcp://localhost:4848")
Ts_plant = int(os.getenv("TS_PLANT", 1))