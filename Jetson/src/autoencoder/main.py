import csv
import time
import numpy as np
from pubsub_redis import PubSubRedis 
from AE_buffer import *
from params import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_data_file(data, csv_file, fieldnames): 
    with open(csv_file, "a") as file: 
        csv_writer = csv.DictWriter(file, fieldnames)
        info = {
            "t": data[0],
            "dt": data[1], 
            "xn1": data[2], 
            "xn2": data[3],
            "xn3": data[4],
            "xn4": data[5],
            "xf1": data[6], 
            "xf2": data[7], 
            "xf3": data[8], 
            "xf4": data[9]
        }
        csv_writer.writerow(info)


def process_data(AE, x, u, scaler):
    """
    Apply autoencoder or any other data processing logic here.
    """
    ## Noise x (the function recieves a numpy array, a conversion from list to np.array has to be done and later reversed)
    if ADD_NOISE:
        x_np = np.array(x).reshape(1,4)
        x_noised = AE.noise_datapoint(x_np, multiplier_white=1, multiplier_SP=1)
        x_noised = x_noised[0].tolist()
    else: 
        x_np = x
    
    if USE_AUTOENCODER:
        # The point is concatenated to the input (not needed for buffer to work) and then reshaped
        point = np.concatenate((u, x_noised), axis=None).reshape(1,6)
        
        # Fit data to scaling scheme (a np.array and a reshape is needed)
        point_scaled = scaler.transform(np.array(point))
        point_scaled = np.concatenate(point_scaled, axis=None).reshape(1,6)
        
        ## Autoencoder
        AE.buffer(data_point=point_scaled)
        # Convert back
        Y_pred_unscaled = AE.Y_pred_list[-1]
        Y_pred_list = scaler.inverse_transform(Y_pred_unscaled)
        # Select the last element of the buffer (tensor), not regarding the inputs (first two elements) and reshaping
        x_pred = Y_pred_list[-1][2:].reshape(4,)
    else: 
        x_pred = x
    return x_noised, x_pred



if __name__ == "__main__":

    csv_path = "../../results/autoencoder.csv"
    fieldnames = [
        "t", "dt", 
        "xn1", "xn2", 
        "xn3", "xn4", 
        "xf1", "xf2", 
        "xf3", "xf4"
    ]
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

    ## Initialize the autoencoder
    folder='NL_AE_models/'
    noise='saltPepper'
    name='uut_noise2'
    meta = load_metadata('{}{}/'.format(folder, name))
    processor = DataProcessor(seqlen=60)

    # data_dict2 no tiene shuffle
    data_dict2 = processor.process_tanks(noise_power=meta['noise_power'], 
                                         noise_inputs=meta['noise_inputs'], 
                                         shuffle=False,
                                         folder='Tanks_Data/No_Noised_Inputs/', 
                                         type_of_noise=noise, clean_data='data_NL_clean.pkl')
    test_data = data_dict2['test_data']
    clean_data = data_dict2['test_data_preproc']
    print(f"Test data shape: {test_data.shape}")

    scaler = data_dict2['scaler']
    scaler_preproc = data_dict2['scaler_preproc']
    np_clean_data = clean_data.numpy()
    np_clean_data = scaler_preproc.inverse_transform(np_clean_data[:, 0, :])
    clean_data = torch.from_numpy(np_clean_data)
    print(f"Clean data shape: {clean_data.shape}")

    AE = AutoEncoder()

    try: 
        print("//---------------------- Initializing Communication ----------------------//")
        xf = np.zeros(shape=(4,))
        start_time = time.time()
        while True:
            current_time = time.time()
            # Get the latest data from the subscribed channel
            plant_redis = d_redis.data_subs
            x = plant_redis[:4]
            u = plant_redis[4:]

            ## Stop the simulation
            if (x[0], x[1], x[2], x[3], u[0], u[1]) == (1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0): 
                break

            # Check if data is valid before processing
            if x is not None and u is not None and x.size > 0 and u.size > 0:
                # Process the data:
                xn, xf = process_data(AE, x, u, scaler)
                ## Publishing
                d_redis.publish_data(channel_pub, xf)
                print('Published (Redis) to plant_outputs_filtered:')
                print('(x_1, x_2, x_3, x_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(xf[0], xf[1], xf[2], xf[3]))
                print(30*'-')

            ## Event handling!
            d_redis.wait_for_new_data()

            elapsed_time = current_time - start_time
            dt = time.time() - current_time
            ## Store simulation time, delta time, and variables in data array
            data = [
                elapsed_time, dt, 
                xn[0], xn[1], 
                xn[2], xn[3],
                xf[0], xf[1], 
                xf[2], xf[3]
            ]
            write_data_file(data, csv_path, fieldnames)
    finally: 
        d_redis.stop_subscribing()
