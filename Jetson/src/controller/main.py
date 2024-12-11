import time
import csv
import numpy as np
from pubsub_redis import PubSubRedis
from params import *


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
            "t": data[0],
            "dt": data[1], 
            "u1": data[2], 
            "u2": data[3], 
            "ref": data[4]
        }
        csv_writer.writerow(info)
        


if __name__ == "__main__": 

    ## Instanciate Controller:
    controller = Controller(TS_PLANT)

    csv_path = "/results/controller.csv"
    fieldnames = ["t", "dt", "u1", "u2", "ref"]
    with open(csv_path, "w") as file: 
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    d_redis = PubSubRedis(
        host="localhost", 
        port=6379, 
        driver_name="controller_driver"
    )
    d_redis.data_subs = np.zeros(shape=(4,))        ## Initialize the data channel to be subscribed (xf1, xf2, xf3, xf4)
    
    ## Publishing & subscribing channels: 
    channel_sub = "plant_outputs_filtered"
    channel_pub = "plant_inputs"

    ## Start the subscription (receiving data from plant_outputs_filtered)
    d_redis.start_subscribing(channel_sub)

    try: 
        print("//---------------------- Initializing Communication ----------------------//")

        cnt = 0                             # Counter for simulation
        sim_points = 20000                  # Number of simulation points
        ref = 15 * np.ones(4)               # init reference 4-Tanks plant
        u = np.zeros(shape=(2,))            # init manipulated variable

        start_time = time.time()
        while True:
            current_time = time.time()
            # Get the latest data from the subscribed channel
            yf = d_redis.data_subs

            # Check if data is valid before processing
            if yf is not None and yf.size > 0:
                # Process the data:
                controller.integral_control(yf, ref)
                u = np.array([controller.u[0], controller.u[1]])
                # Publishing:
                d_redis.publish_data(channel_pub, u)
                print('Published (Redis) to plant_inputs:')
                print('(u_1, u_2) = ({:.2f}, {:.2f})'.format(u[0], u[1]))
                print('Tracking sample cnt number: {}'.format(cnt))
                print('Reference: ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(ref[0], ref[1], ref[2], ref[3]))
                print(30*'-')
                ## Reference changes in 1/3 and 2/3 of the simulation length
                if cnt == int(sim_points/10): ref = 20 * np.ones(4)
                if cnt == int(2*sim_points/10): ref = 25 * np.ones(4)
                if cnt == int(3*sim_points/10): ref = 5 * np.ones(4)
                if cnt == int(4*sim_points/10): ref = 10 * np.ones(4)
                if cnt == int(5*sim_points/10): ref = 20 * np.ones(4)
                if cnt == int(6*sim_points/10): ref = 35 * np.ones(4)
                if cnt == int(7*sim_points/10): ref = 25 * np.ones(4)
                if cnt == int(8*sim_points/10): ref = 15 * np.ones(4)
                if cnt == int(9*sim_points/10): ref = 30 * np.ones(4)
                ## Stop the simulation after sim_points
                if cnt >= sim_points:
                    ## Stop the simulation: flag to client
                    d_redis.publish_data(channel_pub, 1000.0*np.ones_like(u))
                    cnt = 0
                    break
                else:
                    cnt += 1

            ## Event handling!
            d_redis.wait_for_new_data()

            elpased_time = current_time - start_time
            dt = time.time() - current_time
            ## Store simulation time, delta time, and variables in data array
            data = [elpased_time, dt, u[0], u[1], ref[0]]
            write_data_file(data, csv_path, fieldnames)
    finally: 
        d_redis.stop_subscribing()
