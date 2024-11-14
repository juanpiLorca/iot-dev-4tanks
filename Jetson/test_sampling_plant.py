import threading
import time
from opcua import Client

class SubHandler:
    """Subscription Handler to receive events from server for multiple subscriptions"""

    def __init__(self):
        # Dictionary to store latest values of each node
        self.latest_data = {}
        # Lock to ensure thread safety when accessing latest_data
        self.data_lock = threading.Lock()
        # Event to indicate when new data is available
        self.new_data_event = threading.Event()

    def datachange_notification(self, node, val, data):
        """Handles data changes and stores them in a thread-safe manner"""
        node_id = node.nodeid.Identifier  # Using node ID as a key; adjust as needed
        with self.data_lock:
            self.latest_data[node_id] = val
        self.new_data_event.set()

    def get_latest_value(self, node_id):
        """Retrieve the latest value for a specific node in a thread-safe manner"""
        with self.data_lock:
            return self.latest_data.get(node_id)

    def get_all_latest_values(self):
        """Retrieve all latest values in a thread-safe manner"""
        with self.data_lock:
            return self.latest_data.copy()

if __name__ == "__main__":
    url = "opc.tcp://192.168.0.122:4848"
    client = Client(url)
    client.connect()

    handler = SubHandler()  
    checking_time = 500 # ms
    client.create_subscription(checking_time, handler)

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

    sub = client.create_subscription(checking_time, handler)
    y1_val = sub.subscribe_data_change(y_1_node)
    y2_val = sub.subscribe_data_change(y_2_node)
    y3_val = sub.subscribe_data_change(y_3_node)
    y4_val = sub.subscribe_data_change(y_4_node)

    try:
        while True:
            # Wait for new data to be available
            initial_time = time.time()
            handler.new_data_event.wait()
            
            # Retrieve the latest values for each node
            y1 = handler.get_latest_value(y_1_node.nodeid.Identifier)
            y2 = handler.get_latest_value(y_2_node.nodeid.Identifier)
            y3 = handler.get_latest_value(y_3_node.nodeid.Identifier)
            y4 = handler.get_latest_value(y_4_node.nodeid.Identifier)
            print(f"Current values: y1={y1}, y2={y2}, y3={y3}, y4={y4}")
            
            # Clear the event to wait for the next data change
            handler.new_data_event.clear()
            final_time = time.time()
            print(f"Time elapsed: {final_time - initial_time} seconds")
    finally:
        client.disconnect()
        print("Client disconnected")

