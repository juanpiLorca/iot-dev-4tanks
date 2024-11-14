import numpy as np
import csv
import time
import threading
from opcua import Client 
from pubsub_redis import PubSubRedis
from params import *

class SubHandler(object):
    """Subscription Handler. Receives events from the server for a subscription"""

    def __init__(self):
        ## Dictionary to store lastets values of each node: {y1, y2, y3, y4}
        self.latest_data = {}
        self.data_lock = threading.Lock()
        self.new_data_event = threading.Event()

    def datachange_notification(self, node, val, data):
        """Handles data changes and stores them in a thread-safe manner"""
        node_id = node.nodeid.Identifier  # Using node ID as a key
        with self.data_lock:
            self.latest_data[node_id] = val
        self.new_data_event.set()

    def get_latest_value(self, node_id):
        """Retrieve the latest value for a specific node in a thread-safe manner"""
        with self.data_lock:
            return self.latest_data.get(node_id)



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

    csv_path = "/workspace/results/plant_client.csv"
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

    ## Handler for event handling: 
    handler = SubHandler()
    ## Still having the problem of not being able to sample the plant as plant sampling time
    ## by event handling. ==> Check test_sampling_plant.py script to see that, even by forcing 
    ##                        the plant to change by applying mv. the plant states change in the 
    ##                        server by a really small amount that triggers the event handler in 
    ##                        less than 1 second.
    sub = client.create_subscription(int(TS_PLANT*1000), SubHandler())
    y1_val = sub.subscribe_data_change(y_1_node)
    y2_val = sub.subscribe_data_change(y_2_node)
    y3_val = sub.subscribe_data_change(y_3_node)
    y4_val = sub.subscribe_data_change(y_4_node)

    d_redis = PubSubRedis(
        host="localhost", 
        port=6379, 
        driver_name="plant_client_driver"
    )
    d_redis.data_subs = np.zeros(shape=(2,))        ## Initialize the data channel to be subscribed (u1, u2)
    
    ## Define the channels for pub/sub
    channel_pub = "plant_outputs"
    channel_sub = "plant_inputs"

    ## Start the subscription (receiving data from plant_inputs)
    d_redis.start_subscribing(channel_sub)
    
    try:   
        print("//---------------------- Initializing Communication ----------------------//")
        u = np.array([0.40, 0.30])          # INITIAL CONDITION 
        d_redis.data_subs = u
        start_time = time.time()
        while True: 
            current_time = time.time()
            handler.new_data_event.wait()   

            # Get the latest data from the subscribed channel 
            u = d_redis.data_subs 

            ## Stop the simulation: flag to autoencoder
            if (u[0], u[1]) == (1000.0, 1000.0): 
                y_flag = 1000.0*np.ones(6)
                d_redis.publish_data(channel_pub, y_flag)
                break
            
            # Check if data is valid before processing 
            if u is not None and u.size > 0: 

                # Publish manipulated variable to plant server
                u_1_node.set_value(u[0])
                u_2_node.set_value(u[1])
                print('Published to plant: (u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
                print(30*'-')

                # Get data from plant server 
                y = np.array([
                        handler.get_latest_value(y_1_node.nodeid.Identifier),
                        handler.get_latest_value(y_2_node.nodeid.Identifier),
                        handler.get_latest_value(y_3_node.nodeid.Identifier),
                        handler.get_latest_value(y_4_node.nodeid.Identifier)
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
        client.disconnect()


        