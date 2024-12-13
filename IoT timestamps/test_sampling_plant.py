import numpy as np
import csv
import time
import threading
from opcua import Client 
from params import *

class SubHandler(object):
    """Subscription Handler to receive events from server for multiple subscriptions"""

    def __init__(self):
        self.latest_data = {}
        self.data_lock = threading.Lock()
        self.new_data_event = threading.Event()

    def datachange_notification(self, node, val, data):
        """Handles data changes and stores them in a thread-safe manner"""
        node_id = node.nodeid.Identifier  
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
        

class Controller():
    def __init__(self, Ts):
        self.Ts = Ts
        self.error = np.zeros((4,))
        self.u = np.zeros((2,))

        ## Define matrices
        # Use LQR to design the state feedback gain matrix K
        self.K = np.array([
            [0.7844,    0.1129,   -0.0768,    0.5117],
            [0.0557,    0.7388,    0.5409,   -0.0397]   
        ])
        self.K_i = np.array([
            [0.9107,   -0.0497,    0.1049,    0.0037,    0.0039,   -0.0004],
            [-0.0475,    1.4749,    0.0072,    0.1484,    0.0002,    0.0159]   
        ]) 
        
    ## "Real time" integration
    def closed_loop(self, x_in, ref = np.zeros((4,))):
        # Calculate control input u = -Kx
        self.u = -self.K @ (x_in - ref)

    ## Open loop dynamics: input is u (2x1) and output is x (4x1)
    def open_loop(self, u):
        ## Input
        self.u = u

    ## Control with integral squeme
    def integral_control(self, x_in, ref = np.zeros((4,))):
        ## Error
        e = x_in - ref
        self.error += e * self.Ts
        ## Calculate control input u = -K_i(x + e)
        x_concat = np.hstack((x_in, self.error[:2])) # Error only for x_0 and x_1
        self.u = -self.K_i @ x_concat



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

    csv_path = "data_test/plant_client.csv"
    fieldnames = ["t", "dt", "x1", "x2", "x3", "x4", "u1", "u2"]
    with open(csv_path, "w") as file: 
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    print("//---------------------- Plant Client ----------------------//")
     ## Instanciate Controller:
    controller = Controller(TS_PLANT)

    ## Instanciate Plant Client: 
    client = Client(PLANT_SERVER_IP)
    client.connect()
    print(f"Client connected to plant server at: {PLANT_SERVER_IP}")

    ## Search for particular nodes
    root = client.get_root_node()
    objects = client.get_objects_node()
    tanks = objects.get_child(['2:QuadrupleTanks'])
    inputs_folder = tanks.get_child(['2:Inputs'])
    flags_floder = tanks.get_child(['2:Flags'])
    perturbations_folder = tanks.get_child(['2:Perturbations'])
    outputs_folder = tanks.get_child(['2:Outputs'])

    ## Access the variables
    ## Inputs
    u_1_node = inputs_folder.get_child(['2:u_1'])
    u_2_node = inputs_folder.get_child(['2:u_2'])
    in_pkg_node = inputs_folder.get_child(['2:in_pkg'])
    ## Flags
    flag_node1 = flags_floder.get_child(['2:flag2start'])
    flag_node2 = flags_floder.get_child(['2:flag2stop'])
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

    ## Define the channels for pub/sub
    channel_pub = "plant_outputs"
    channel_sub = "plant_inputs"

    try:   
        print("//---------------------- Initializing Communication ----------------------//")

        u = np.array([0.40, 0.30])          # INITIAL CONDITION: manipulated variable 
        pkg_timer = 0                       # Counter for simulation

        ref = 15 * np.ones(4)               # init reference 4-Tanks plant
        cnt = 0
        sim_points = 2000

        start_time = time.time()
        flag_node1.set_value(1)             # Start taking timestamps in Raspberry Pi
        while True: 
            current_time = time.time()
            print("waiting for new data...")
            handler.new_data_event.wait()   
            print(f"new data received, dt := {time.time() - current_time}")

            # Get data from plant server 
            y = np.array([
                    y_1_node.get_value(),
                    y_2_node.get_value(),
                    y_3_node.get_value(),
                    y_4_node.get_value()
            ])

            controller.integral_control(y, ref)
            u = np.array([controller.u[0], controller.u[1]])
            # Publish manipulated variable to plant server
            u_1_node.set_value(u[0])
            u_2_node.set_value(u[1])
            in_pkg_node.set_value(pkg_timer+1)
            pkg_timer += 1
            print('Published to plant: (u_1, u_2, pkg) = ({:.2f}, {:.2f}, {:.2f})'.format(u[0], u[1], pkg_timer))
            print(30*'-')

            ## Reference changes in 1/3 and 2/3 of the simulation length
            if cnt == int(sim_points/10): ref = 20 * np.ones(4)
            if cnt == int(2*sim_points/10): ref = 25 * np.ones(4)
            if cnt == int(3*sim_points/10): ref = 5 * np.ones(4)
            ## Stop the simulation: flag to client
            if cnt > sim_points:
                flag_node2.set_value(1)     ## Set the flag to stop the simulation in Raspberry Pi
                cnt = 0
                break
            else:
                cnt += 1

            ## Clear the event to wait for the next data change: counter flag
            handler.new_data_event.clear()

            elapsed_time = current_time - start_time
            dt = time.time() - current_time

            ## Store simulation time, delta time, and variables in data array
            data = [elapsed_time, dt, y[0], y[1], y[2], y[3], u[0], u[1]]
            write_data_file(data, csv_path, fieldnames)
    finally: 
        client.disconnect()


        