import numpy as np
import csv
import time
from opcua import Client 
from pubsub_redis import PubSubRedis
from util import UtilTimer
from params import *


def write_data_file(data, csv_file, fieldnames): 
    with open(csv_file, "a") as file: 
        csv_writer = csv.DictWriter(file, fieldnames)
        info = {
            "t": data[0],   # Simulation time
            "dt": data[1],  # Delta time (time for each loop iteration)
            "x1": data[2], 
            "x2": data[3], 
            "x3": data[4], 
            "x4": data[5], 
            "u1": data[6], 
            "u2": data[7]
        }
        csv_writer.writerow(info)


if __name__ == "__main__": 

    csv_path = "results/plant_client.csv"
    fieldnames = ["t", "dt", "x1", "x2", "x3", "x4", "u1", "u2"]
    with open(csv_path, "w") as file: 
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    ## Instanciate Plant Client: 
    client = Client(PLANT_SERVER_IP)
    client.connect()

    ## Search for particular nodes
    root = client.get_root_node()
    objects = client.get_objects_node()
    tanks = objects.get_child(['2:QuadrupleTanks'])
    inputs_folder = tanks.get_child(['2:Inputs'])
    perturbations_folder = tanks.get_child(['2:Perturbations'])
    outputs_folder = tanks.get_child(['2:Outputs'])

    ## Access the variables
    u_1_node = inputs_folder.get_child(['2:u_1'])
    u_2_node = inputs_folder.get_child(['2:u_2'])
    y_1_node = outputs_folder.get_child(['2:y_1'])
    y_2_node = outputs_folder.get_child(['2:y_2'])
    y_3_node = outputs_folder.get_child(['2:y_3'])
    y_4_node = outputs_folder.get_child(['2:y_4'])

    d_redis = PubSubRedis(
        host="localhost", 
        port=6379, 
        driver_name="plant_client_driver"
    )

    ## Define the channels for pub/sub
    channel_pub = "plant_outputs"
    channel_sub = "plant_inputs"

    ## Start the subscription (receiving data from plant_inputs)
    d_redis.start_subscribing(channel_sub)

    ## Plant sample time: 
    plant_timer = UtilTimer(Ts_PLANT, "Sampling Plant Timer")
    plant_timer.start()
    
    try: 
        u = np.array([0.40, 0.30])          # INITIAL CONDITION 
        d_redis.data_subs = u
        start_time = time.time()
        while True: 
            current_time = time.time()
            plant_timer.wait()

            # Get the latest data from the subscribed channel 
            u = d_redis.data_subs 
            
            # Check if data is valid before processing 
            if u is not None and u.size > 0: 

                # Publish manipulated variable to plant server
                u_1_node.set_value(u[0])
                u_2_node.set_value(u[1])
                print('Published to plant: (u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
                print(30*'-')

                # Get data from plant server 
                y = np.array([
                    y_1_node.get_value(), 
                    y_2_node.get_value(), 
                    y_3_node.get_value(), 
                    y_4_node.get_value()
                ])
                y_pub = np.concatenate([y, np.array([u[0], u[1]])])
                ## Publishing
                d_redis.publish_data(channel_pub, y_pub)

            elapsed_time = current_time - start_time
            dt = time.time() - current_time
            ## Store simulation time, delta time, and variables in data array
            data = [elapsed_time, dt, y[0], y[1], y[2], y[3], u[0], u[1]]
            write_data_file(data, csv_path, fieldnames)
    finally: 
        d_redis.stop_subscribing()
        plant_timer.stop()


        