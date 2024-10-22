import redis
import threading
import json
import numpy as np
import queue

class PubRedis: 
    def __init__(self, host, port, driver_name):
        self.driver_name = driver_name
        self.r_client = redis.Redis(
            host=host,
            port=port,
            charset="utf-8",
            decode_responses=True,
            db=0
        )
        self.publisher_thread  = None
        self.publish_queue = queue.Queue()
        self.new_data_event = threading.Event()
        self.force_shutdown = True

    def publish_thread(self, channel):
        while True:
            if self.force_shutdown: 
                break
            # Wait for the new data event to be set
            self.new_data_event.wait()
            if not self.publish_queue.empty():
                var = self.publish_queue.get()
                print(f"Publishing updated value: {json.dumps(var.tolist())}")
                self.r_client.publish(channel, json.dumps(var.tolist()))
                self.new_data_event.clear()  # Reset event after publishing

    def start_publishing(self, channel):
        self.publisher_thread = threading.Thread(
            target=self.publish_thread, args=(channel,)
        )
        self.publisher_thread.start()

    def update_value(self, var):
        self.publish_queue.put(var)
        # Signal that new data is ready to be published
        self.new_data_event.set() 

    def stop_publishing(self): 
        self.force_shutdown = True
        self.publisher_thread.join()


class SubRedis:
    def __init__(self, host, port, driver_name):
        self.driver_name = driver_name
        self.r_client = redis.Redis(
            host=host,
            port=port,
            charset="utf-8",
            decode_responses=True,
            db=0
        )
        self.subscriber_thread = None
        self.force_shutdown = True

    def subscribe_thread(self, channel):
        pubsub = self.r_client.pubsub()
        pubsub.subscribe(channel)
        print(f"Subscribed to channel {channel}")

        for msg in pubsub.listen():
            if self.force_shutdown: 
                break

            if msg['type'] == 'message':
                self.data_subs = np.array(json.loads(msg["data"]))
                print(f"Received data: {self.data_subs}")
            else:
                print(f"Non-message event: {msg}")

    def start_subscribing(self, channel):
        self.subscriber_thread = threading.Thread(
            target=self.subscribe_thread, args=(channel,)
        )
        self.subscriber_thread.start() 
    
    def stop_subscribing(self): 
        self.force_shutdown = True
        self.subscriber_thread.join()


# Redis client
class PubSubRedis:
    def __init__(self, host, port, driver_name):
        self.driver_name = driver_name
        self.r_client = redis.Redis(
            host=host, 
            port=port, 
            charset="utf-8", 
            decode_responses=True
        )
        self.subscriber_thread = None
        self.publisher_thread = None
        self.data_subs = None
        self.publish_queue = queue.Queue()
        self.new_data_event = threading.Event()
        self.force_shutdown = False

    def subscribe_thread(self, channel):
        pubsub = self.r_client.pubsub()
        pubsub.subscribe(channel)
        print(f"Subscribed to channel {channel}")
        
        for msg in pubsub.listen():
            if self.force_shutdown: 
                break

            if msg['type'] == 'message':
                # New data received, update it and set the event
                self.data_subs = np.array(json.loads(msg["data"]))
                print(f"Received new data: {self.data_subs}")
                # Trigger event to process the new data
                self.new_data_event.set()
    
    def publish_data(self, channel, data):
        #print(f"Publishing processed data: {data}")
        self.r_client.publish(channel, json.dumps(data.tolist()))
        self.new_data_event.clear()

    def start_subscribing(self, channel):
        self.subscriber_thread = threading.Thread(
            target=self.subscribe_thread, args=(channel,)
        )
        self.subscriber_thread.start()

    def wait_for_new_data(self): 
        ## Event handling: must be on the end of every loop
        self.new_data_event.wait()

    def stop_subscribing(self): 
        self.force_shutdown = True
        self.subscriber_thread.join()

