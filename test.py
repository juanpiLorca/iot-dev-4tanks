from opcua import Client

# Set up the OPC UA server address
url = "opc.tcp://192.168.0.122:4840"  # Replace 4840 with your OPC UA server port if different
client = Client(url)

try:
    # Connect to the OPC UA server
    client.connect()
    print(f"Connected to OPC UA server at {url}")

    # Read a variable from the server
    node_id = "ns=2;i=2"  # Replace this with the actual node ID you want to read
    node = client.get_node(node_id)
    value = node.get_value()
    print(f"Value of node {node_id}: {value}")

    # Write a value to the node (if writable)
    new_value = 42  # Replace this with the desired value
    node.set_value(new_value)
    print(f"Updated node {node_id} to value: {new_value}")

except Exception as e:
    print("Failed to connect or perform operations:", e)
finally:
    # Disconnect from the server
    client.disconnect()
    print("Disconnected from OPC UA server")
