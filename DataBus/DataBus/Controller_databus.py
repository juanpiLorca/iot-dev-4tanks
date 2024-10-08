import time
import csv
import numpy as np
from pubsub_redis import PubSubRedis

def write_data_file(data, csv_file, fieldnames): 
    with open(csv_file, "a") as file: 
        csv_writer = csv.DictWriter(file, fieldnames)
        info = {
            "t": data[0],
            "dt": data[1], 
            "u1": data[2], 
            "u2": data[3], 
            "u3": data[4], 
            "u4": data[5]
        }
        csv_writer.writerow(info)

def control_action(x): 
    u = x * (1/2)
    return u

if __name__ == "__main__": 

    csv_path = "results/controller.csv"
    fieldnames = ["t", "dt", "u1", "u2", "u3", "u4"]
    with open(csv_path, "w") as file: 
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    d_redis = PubSubRedis(
        host="localhost", 
        port=6379, 
        driver_name="controller_driver"
    )

    ## Publishing & subscribing channels: 
    channel_sub = "plant_outputs_filtered"
    channel_pub = "plant_inputs"
    d_redis.start_subscribing(channel_sub)

    u = np.zeros(shape=(4,))
    start_time = time.time()
    while True:
        current_time = time.time()
        # Get the latest data from the subscribed channel
        xf = d_redis.data_subs

        # Check if data is valid before processing
        if xf is not None and xf.size > 0:
            # Process the data:
            u = control_action(xf)
            # Publish the processed data
            d_redis.publish_data(channel_pub, u)

        ## Event handling!
        d_redis.wait_for_new_data()

        elpased_time = current_time - start_time
        dt = time.time() - current_time
        data = [elpased_time, dt, u[0], u[1], u[2], u[3]]
        write_data_file(data, csv_path, fieldnames)
