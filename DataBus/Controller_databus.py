import numpy as np
from pubsub_redis import PubSubRedis

def control_action(x): 
    u = x * np.random.normal(loc=0, scale=1, size=x.shape[0])
    return u

if __name__ == "__main__": 

    d_redis = PubSubRedis(
        host="localhost", 
        port=6379, 
        driver_name="controller_driver"
    )

    ## Publishing & subscribing channels: 
    channel_sub = "plant_outputs_filtered"
    channel_pub = "plant_inputs"
    d_redis.start_subscribing(channel_sub)

    while True:
        # Get the latest data from the subscribed channel
        xf = d_redis.data_subs

        # Check if data is valid before processing
        if xf is not None and xf.size > 0:
            # Process the data:
            u = control_action(xf)
            # Publish the processed data
            d_redis.publish_data(channel_pub, u)

        ## Event handling!
        d_redis.wait_for_new_data()
