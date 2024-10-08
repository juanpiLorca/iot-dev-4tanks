import csv
import time
import numpy as np
from pubsub_redis import PubSubRedis 

def write_data_file(data, csv_file, fieldnames): 
    with open(csv_file, "a") as file: 
        csv_writer = csv.DictWriter(file, fieldnames)
        info = {
            "t": data[0],
            "dt": data[1], 
            "xf1": data[2], 
            "xf2": data[3], 
            "xf3": data[4], 
            "xf4": data[5]
        }
        csv_writer.writerow(info)

def process_data(xf):
    """
    Apply autoencoder or any other data processing logic here.
    For now, we'll just simulate with a transformation.
    """
    # Example transformation, replace with actual autoencoder logic
    processed_xf = xf * 2
    return processed_xf

if __name__ == "__main__":

    csv_path = "results/autoencoder.csv"
    fieldnames = ["t", "dt", "xf1", "xf2", "xf3", "xf4"]
    with open(csv_path, "w") as file: 
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    ## Initialize the Redis client
    d_redis = PubSubRedis(
        host="localhost",
        port=6379,
        driver_name="autoencoder_driver"
    )
    
    ## Define the channels for pub/sub
    channel_pub = "plant_outputs_filtered"
    channel_sub = "plant_outputs"

    ## Start the subscription (receiving data from plant_outputs)
    d_redis.start_subscribing(channel_sub)

    t_delay = 0.5                   # [s]
    xf = np.zeros(shape=(4,))
    start_time = time.time()
    while True:
        current_time = time.time()
        # Get the latest data from the subscribed channel
        x = d_redis.data_subs

        # Check if data is valid before processing
        if x is not None and x.size > 0:
            # Process the data:
            xf = process_data(x)
            # Publish the processed data (delayed)
            time.sleep(t_delay)
            d_redis.publish_data(channel_pub, xf)

        ## Event handling!
        d_redis.wait_for_new_data()

        elapsed_time = current_time - start_time
        dt = time.time() - current_time
        data = [elapsed_time, dt, xf[0], xf[1], xf[2], xf[3]]
        write_data_file(data, csv_path, fieldnames)
