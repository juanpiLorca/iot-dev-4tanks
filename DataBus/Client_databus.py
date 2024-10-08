import numpy as np
import csv
import time
from pubsub_redis import PubRedis
from util import UtilTimer


def write_data_file(data, csv_file, fieldnames): 
    with open(csv_file, "a") as file: 
        csv_writer = csv.DictWriter(file, fieldnames)
        info = {
            "t": data[0],   # Simulation time
            "dt": data[1],  # Delta time (time for each loop iteration)
            "x1": data[2], 
            "x2": data[3], 
            "x3": data[4], 
            "x4": data[5]
        }
        csv_writer.writerow(info)


if __name__ == "__main__": 

    csv_path = "results/plant_client.csv"
    fieldnames = ["t", "dt", "x1", "x2", "x3", "x4"]
    with open(csv_path, "w") as file: 
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    d_redis = PubRedis(
        host="localhost", 
        port=6379, 
        driver_name="plant_client_driver"
    )

    ## Plant sample time: 
    Ts = 1
    plant_timer = UtilTimer(Ts, "Sampling Timer")
    plant_timer.start()

    ## Only for publishing: 
    channel_pub = "plant_outputs"
    d_redis.start_publishing(channel_pub)
    
    x = np.zeros(shape=(4,))
    start_time = time.time()
    while True: 
        current_time = time.time()
        plant_timer.wait()

        x += 0.01
        d_redis.update_value(x)

        ## Calculate elapsed simulation time
        elapsed_time = current_time - start_time
        dt = time.time() - current_time

        ## Store simulation time, delta time, and variables in data array
        data = [elapsed_time, dt, x[0], x[1], x[2], x[3]]
        write_data_file(data, csv_path, fieldnames)


        