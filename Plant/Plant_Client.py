from opcua import Client
import time, sys, os
import numpy as np
from NL_QuadrupleTank import NL_QuadrupleTank
## Import from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from params import *

if __name__ == '__main__':
    ## Instanciate Plant
    x0=[12.4, 12.7, 1.8, 1.4]
    plant = NL_QuadrupleTank(x0=x0, Ts=Ts_plant)

    ## Create a client instance and connect to the server
    client = Client(plant_server_ip) 
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

    while True:
        start_time = time.time()
        ## Get the values from the controller and update the plant
        
        ## Plant step
        # u = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
        # u = np.array([0.5,0.5])
        # u = np.array([plant.u[0], plant.u[1]])
        u = np.array([u_1_node.get_value(), u_2_node.get_value()])
        plant.step(u)

        # Update the server variables
        # u_1_node.set_value(plant.u[0])
        # u_2_node.set_value(plant.u[1])

        # Read the server variables
        y_1_node.set_value(plant.x[0])
        y_2_node.set_value(plant.x[1])
        y_3_node.set_value(plant.x[2])
        y_4_node.set_value(plant.x[3])
        print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(plant.x[0], plant.x[1], plant.x[2], plant.x[3]))
        print('(u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
        print(30*'-')

        # Add a delay
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 - elapsed_time))
