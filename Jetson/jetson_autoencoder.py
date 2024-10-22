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

def process_data(xf, t_delay):
    """
    Apply autoencoder or any other data processing logic here.
    For now, we'll just simulate with a buffer.
    """
    # Example transformation, replace with actual autoencoder logic
    time.sleep(t_delay)
    processed_xf = xf[:4]       # JUST RETURNING THE STATES!
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

    try: 
        t_delay = 0.1                   # to simulate autoencoder processing time [s]
        yf = np.zeros(shape=(4,))
        start_time = time.time()
        while True:
            current_time = time.time()
            # Get the latest data from the subscribed channel
            y = d_redis.data_subs

            # Check if data is valid before processing
            if y is not None and y.size > 0:
                # Process the data:
                yf = process_data(y, t_delay)
                ## Publishing
                d_redis.publish_data(channel_pub, yf)
                print('Published (Redis) to plant_outputs_filtered:')
                print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(yf[0], yf[1], yf[2], yf[3]))
                print(30*'-')

            ## Event handling!
            d_redis.wait_for_new_data()

            elapsed_time = current_time - start_time
            dt = time.time() - current_time
            ## Store simulation time, delta time, and variables in data array
            data = [elapsed_time, dt, yf[0], yf[1], yf[2], yf[3]]
            write_data_file(data, csv_path, fieldnames)
    finally: 
        d_redis.stop_subscribing()
