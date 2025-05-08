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
        ## Dictionary to store lastets values of each node: {y1, y2, y3, y4, pkg}
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

    def get_all_latest_values(self):
        """Retrieve all latest values in a thread-safe manner"""
        with self.data_lock:
            return self.latest_data.copy()



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



class Input2Client: 
    """Class to handle OPC UA client for the plant server"""

    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.client = Client(server_ip)
        self.handler = SubHandler()
        self.data_lock = threading.Lock()

        self.root = None
        self.objects = None
        self.plant = None
        self.inputs_folder = None
        self.perturbations_folder = None
        self.outputs_folder = None

        ## Defined for 4-Tank System
        self.u_1_node = None
        self.u_2_node = None
        self.in_pkg_node = None         # Package counter input for event handling
        self.out_pkg_node = None        # Package counter output for event handling
        self.y_1_node = None    
        self.y_2_node = None
        self.y_3_node = None
        self.y_4_node = None

        self.handler = SubHandler()     # Event handler for data changes
        self.sub = None
        self.pkg_change = None     


    def connect(self):
        """Connect to the OPC UA server"""
        self.client.connect()

    def disconnect(self):
        """Disconnect from the OPC UA server"""
        self.client.disconnect()

    def get_variables(self): 
        """Get the variables from the server"""
        self.root = self.client.get_root_node()
        objects = self.client.get_objects_node()
        tanks = objects.get_child(['2:QuadrupleTanks'])
        inputs_folder = tanks.get_child(['2:Inputs'])
        self.perturbations_folder = tanks.get_child(['2:Perturbations'])
        outputs_folder = tanks.get_child(['2:Outputs'])

        ## Access the variables
        ## Inputs 
        self.u_1_node = inputs_folder.get_child(['2:u_1'])
        self.u_2_node = inputs_folder.get_child(['2:u_2'])
        self.in_pkg_node = inputs_folder.get_child(['2:in_pkg'])
        ## Outputs
        self.y_1_node = outputs_folder.get_child(['2:y_1'])
        self.y_2_node = outputs_folder.get_child(['2:y_2'])
        self.y_3_node = outputs_folder.get_child(['2:y_3'])
        self.y_4_node = outputs_folder.get_child(['2:y_4'])
        self.out_pkg_node = outputs_folder.get_child(['2:out_pkg'])

    def get_latest_data(self):
        """Get the latest data from the server"""



if __name__ == "__main__": 

    csv_path = "/results/plant_client.csv"
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
    ## Inputs 
    u_1_node = inputs_folder.get_child(['2:u_1'])
    u_2_node = inputs_folder.get_child(['2:u_2'])
    in_pkg_node = inputs_folder.get_child(['2:in_pkg'])
    ## Outputs
    y_1_node = outputs_folder.get_child(['2:y_1'])
    y_2_node = outputs_folder.get_child(['2:y_2'])
    y_3_node = outputs_folder.get_child(['2:y_3'])
    y_4_node = outputs_folder.get_child(['2:y_4'])
    out_pkg_node = outputs_folder.get_child(['2:out_pkg'])

    ## Handler for event handling: 
    handler = SubHandler()
    ## Still having the problem of not being able to sample the plant as plant sampling time
    ## by event handling. ==> Check test_sampling_plant.py script to see that, even by forcing 
    ##                        the plant to change by applying mv. the plant states change in the 
    ##                        server by a really small amount that triggers the event handler in 
    ##                        less than 1 second.
    sub = client.create_subscription(int(TS_PLANT*1000), handler)
    pkg_change = sub.subscribe_data_change(out_pkg_node)

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
    print(f"Subscribing to channel: {channel_sub}")
    
    try:   
        print("//---------------------- Initializing Communication ----------------------//")

        u = np.array([0.40, 0.30])          # INITIAL CONDITION: manipulated variable 
        d_redis.data_subs = u               # Initialize the data channel to be subscribed (u1, u2)
        pkg_timer = 0                       # Counter for simulation

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
                in_pkg_node.set_value(pkg_timer+1)
                pkg_timer += 1
                print('Published to plant: (u_1, u_2, inpkg) = ({:.2f}, {:.2f}, {:.2f})'.format(u[0], u[1], pkg_timer))
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

            ## Clear the event to wait for the next data change: timer in plant_client (Raspberry Pi)
            handler.new_data_event.clear()

            elapsed_time = current_time - start_time
            dt = time.time() - current_time
            ## Store simulation time, delta time, and variables in data array
            data = [elapsed_time, dt, y[0], y[1], y[2], y[3], u[0], u[1]]
            write_data_file(data, csv_path, fieldnames)
    finally: 
        d_redis.stop_subscribing()
        client.disconnect()

