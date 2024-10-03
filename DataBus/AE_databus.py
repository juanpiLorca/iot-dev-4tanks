from pubsub_redis import PubSubRedis #

def process_data(xf):
    """
    Apply autoencoder or any other data processing logic here.
    For now, we'll just simulate with a transformation.
    """
    # Example transformation, replace with actual autoencoder logic
    processed_xf = xf * 2
    return processed_xf

if __name__ == "__main__":

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

    while True:
        # Get the latest data from the subscribed channel
        x = d_redis.data_subs

        # Check if data is valid before processing
        if x is not None and x.size > 0:
            # Process the data:
            xf = process_data(xf)
            # Publish the processed data
            d_redis.publish_data(channel_pub, xf)

        ## Event handling!
        d_redis.wait_for_new_data()
