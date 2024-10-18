from opcua import Client
import time
import numpy as np
from NL_QuadrupleTank import NL_QuadrupleTank
from params import *
from util import *

if __name__ == '__main__':
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
    u_1_node = inputs_folder.get_child(['2:u_1'])
    u_2_node = inputs_folder.get_child(['2:u_2'])
    y_1_node = outputs_folder.get_child(['2:y_1'])
    y_2_node = outputs_folder.get_child(['2:y_2'])
    y_3_node = outputs_folder.get_child(['2:y_3'])
    y_4_node = outputs_folder.get_child(['2:y_4'])

    ## Plant sample time: 
    plant_timer = UtilTimer(Ts_PLANT, "Sampling Plant Timer")
    plant_timer.start()
    
    try:
        start_time = time.time()
        while True:
            plant_timer.wait()

            ## Get the values from the controller and update the plant
            u = np.array([u_1_node.get_value(), u_2_node.get_value()])
            ## Plant step
            plant.step(u)

            ## Read the server variables
            y_1_node.set_value(plant.x[0])
            y_2_node.set_value(plant.x[1])
            y_3_node.set_value(plant.x[2])
            y_4_node.set_value(plant.x[3])
            print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(plant.x[0], plant.x[1], plant.x[2], plant.x[3]))
            print('(u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
            print(30*'-')
    finally: 
        plant_timer.stop()