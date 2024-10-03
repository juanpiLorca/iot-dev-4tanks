import numpy as np
import time
from pubsub_redis import PubRedis

if __name__ == "__main__": 

    d_redis = PubRedis(
        host="localhost", 
        port=6379, 
        driver_name="plant_client_driver"
    )

    ## Only for publishing: 
    channel_pub = "plant_outputs"
    d_redis.start_publishing(channel_pub)

    x = np.zeros(shape=(4,))
    i = 0
    while True: 
        x += 0.01
        d_redis.update_value(x)
        time.sleep(1)
        