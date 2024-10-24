from opcua import Client
from opcua import ua
import sys

def test_opc_connection(server_url):
    try:
        # Initialize the client with the OPC server URL
        client = Client(server_url)
        
        # Connect to the server
        client.connect()
        print(f"Successfully connected to OPC server: {server_url}")
        
        # Verify if the server is responding by reading its server state
        server_state = client.get_node("i=2259").get_value()  # 'i=2259' is the NodeId for ServerState
        print(f"Server state: {server_state}")
        
        # Optionally, browse root folder to check for nodes
        objects_node = client.get_objects_node()
        print("Browsing OPC server objects:")
        for child in objects_node.get_children():
            print(child)

        return True

    except Exception as e:
        print(f"Failed to connect to OPC server: {e}")
        return False

    finally:
        # Always disconnect from the server
        try:
            client.disconnect()
            print("Disconnected from the OPC server.")
        except:
            pass

if __name__ == "__main__":
    success = test_opc_connection("opc.tcp://localhost:4848")
    print(success)