import time
import numpy as np
from util import *
from params import *
from plant import NL_QuadrupleTank
from opcua import Client


def main(): 
    ## Sleep 5 seconds to allow the server to start
    time.sleep(5)
    ## Instanciate Plant
    x0=[12.4, 12.7, 1.8, 1.4]
    plant = NL_QuadrupleTank(x0=x0, Ts=TS_PLANT)

    ## Create a client instance and connect to the server
    client = Client(PLANT_SERVER_IP) 
    client.connect()

    ## Search for particular nodes
    objects = client.get_objects_node()
    tanks = objects.get_child(['2:QuadrupleTanks'])
    inputs_folder = tanks.get_child(['2:Inputs'])
    outputs_folder = tanks.get_child(['2:Outputs'])

    ## Access the variables
    u_1_node = inputs_folder.get_child(['2:u_1'])
    u_2_node = inputs_folder.get_child(['2:u_2'])
    y_1_node = outputs_folder.get_child(['2:y_1'])
    y_2_node = outputs_folder.get_child(['2:y_2'])
    y_3_node = outputs_folder.get_child(['2:y_3'])
    y_4_node = outputs_folder.get_child(['2:y_4'])

    ## Plant sample time: 
    plant_timer = UtilTimer(TS_PLANT, "Sampling Plant Timer")
    plant_timer.start()

    ## Default mainpulated variables:
    u_data = np.load('data/u.npy')
    num_samples = u_data.shape[1]
    
    try:    
        for n in range(num_samples):  
            plant_timer.wait()

            ## Get the values from the controller and update the plant
            u = u_data[:, n]
            ## Plant step
            plant.step(u)

            ## Read the server variables
            u_1_node.set_value(u[0])
            u_2_node.set_value(u[1])
            y_1_node.set_value(plant.x[0])
            y_2_node.set_value(plant.x[1])
            y_3_node.set_value(plant.x[2])
            y_4_node.set_value(plant.x[3])
            print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(plant.x[0], plant.x[1], plant.x[2], plant.x[3]))
            print('(u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
            print(30*'-')
    finally: 
        plant_timer.stop()



if __name__ == '__main__':
    main()
