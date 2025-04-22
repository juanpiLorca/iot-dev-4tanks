import time
from params import *
from opcua import Server

class SubHandler(object):
    """
    Subscription Handler. To receive events from server for a subscription
    """
    def datachange_notification(self, node, val, data):
        print(f'{node.get_browse_name().Name}: {val}')
        
    def event_notification(self, event):
        print("Python: New event", event)


class QuadrupleTaknks_Namespace:
    def __init__(self, objects, idx, server):
        self.server = server
        self.objets = objects
        self.idx = idx

        thickener = self.objets.add_folder(self.idx, 'QuadrupleTanks')
        inputs = thickener.add_object(self.idx, 'Inputs')
        outputs = thickener.add_object(self.idx, 'Outputs')

        ## Inputs
        u_1 = inputs.add_variable(self.idx, 'u_1', 0.4)
        u_2 = inputs.add_variable(self.idx, 'u_2', 0.3)
        u_1.set_writable()
        u_2.set_writable()

        ## Outputs
        y_1 = outputs.add_variable(self.idx, 'y_1', 0)
        y_2 = outputs.add_variable(self.idx, 'y_2', 0)
        y_3 = outputs.add_variable(self.idx, 'y_3', 0)
        y_4 = outputs.add_variable(self.idx, 'y_4', 0)
        y_1.set_writable()
        y_2.set_writable()
        y_3.set_writable()
        y_4.set_writable()

        self.inputs = [u_1, u_2]
        self.outputs = [y_1, y_2, y_3, y_4]

    def subscriptions(self):
        handler_inputs = SubHandler()
        handler_outputs = SubHandler()

        sub_inputs = self.server.create_subscription(100, handler_inputs)
        sub_outputs = self.server.create_subscription(100, handler_outputs)

        for to_subscribe in self.inputs:
            sub_inputs.subscribe_data_change(to_subscribe)
        for to_subscribe in self.outputs:
            sub_outputs.subscribe_data_change(to_subscribe)


class OPC_server:
    def __init__(self, address):
        self.server = Server()
        self.server.set_endpoint(address)
        self.server.set_server_name('Quadruple Tank OPC Server')
        self.namespaces = {}
        self.objects = self.server.get_objects_node()

    def new_namespace(self, uri, namespace, name):
        idx = self.server.register_namespace(uri)
        print('New namespace {}'.format(idx))
        namespace = namespace(self.objects, idx, self.server)
        self.namespaces[name] = namespace

    def start(self):
        self.server.start()
        for name, namespace in self.namespaces.items():
            namespace.subscriptions()

        try:
            while True:
                time.sleep(0.001)
        finally:
            self.server.stop()


def main(): 
    server = OPC_server(PLANT_SERVER_IP)
    server.new_namespace(uri='QuadrupleTank', namespace=QuadrupleTaknks_Namespace, name='QuadrupleTank_namespace')
    server.start()


if __name__ == '__main__':
    main()
    
