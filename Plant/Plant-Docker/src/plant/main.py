from opcua import Client
import time
import sys, os
import threading
import numpy as np
import pandas as pd
from NL_QuadrupleTank import NL_QuadrupleTank
from datetime import datetime

## Import from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from params import *


class SubHandler(object):
    def __init__(self):
        self.counter_pkg = None
        self.incoming_timestamp = None
        self.data_lock = threading.Lock()

    def datachange_notification(self, node, val, data):
        """Handles data changes and stores them in a thread-safe manner"""
        with self.data_lock:
            self.counter_pkg = val
            self.incoming_timestamp = datetime.utcnow()
            # Saving outputs into datafile
            df = pd.DataFrame({
                "incoming_timestamp": [self.incoming_timestamp]
            })
            df.to_csv(r'/data/incomming_timestamps.csv', mode="a", index=False, header=False)



if __name__ == '__main__':
    # CSV info
    df = pd.DataFrame(columns=["incoming_timestamp"])
    df.to_csv(r'/data/incomming_timestamps.csv', index=False)
    df = pd.DataFrame(columns=["outgoing_timestamp"])
    df.to_csv(r'/data/outgoing_timestamps.csv', index=False)

    ## Instanciate Plant
    x0=[12.4, 12.7, 1.8, 1.4]
    plant = NL_QuadrupleTank(x0=x0, Ts=Ts_PLANT)

    ## Create a client instance and connect to the server
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

    ## Handler for events:
    handler = SubHandler()
    ## Access incomming package change:
    sub = client.create_subscription(10, handler)
    pkg_change = sub.subscribe_data_change(in_pkg_node)
    ## Timer for outgoing package: 
    pkg_timer = 0

    while True:
        start_time = time.time()
        
        ## First package: t0 + Ts + 2*delta
        u = np.array([u_1_node.get_value(), u_2_node.get_value()])
        plant.step(u)

        ## Update server variables
        y_1_node.set_value(plant.x[0])
        y_2_node.set_value(plant.x[1])
        y_3_node.set_value(plant.x[2])
        y_4_node.set_value(plant.x[3])
        out_pkg_node.set_value(pkg_timer + 1)
        pkg_timer += 1

        outgoing_timestamp = datetime.utcnow()
        ## Second package: t = t0 + Ts

        # Saving outputs into datafile
        df = pd.DataFrame({
            "outgoing_timestamp": [outgoing_timestamp]
        })
        df.to_csv(r'/data/outgoing_timestamps.csv', mode="a", index=False, header=False)
        print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(plant.x[0], plant.x[1], plant.x[2], plant.x[3]))
        print('(u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
        print(30*'-')

        end_time = time.time()
        delta = end_time - start_time
        time.sleep(Ts_PLANT - delta)