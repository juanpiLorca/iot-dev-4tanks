from opcua import Server
import time, sys, os
from datetime import datetime
## Import from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from params import *
class SubHandler(object):

    """
    Subscription Handler. To receive events from server for a subscription
    """
    def __init__(self):
        pass

    def datachange_notification(self, node, val, data):
        print(f'{node.get_browse_name().Name}: {val}')
        pass

    def event_notification(self, event):
        print("Python: New event", event)


class QuadrupleTaknks_Namespace:
    def __init__(self, objects, idx, server):
        self.server = server
        self.objets = objects
        self.idx = idx

        thickener = self.objets.add_folder(self.idx, 'QuadrupleTanks')
        inputs = thickener.add_object(self.idx, 'Inputs')
        disturbances = thickener.add_object(self.idx, 'Perturbations')
        outputs = thickener.add_object(self.idx, 'Outputs')

        ## Disturbances
        # dv_1 = disturbances.add_variable(self.idx, 'disturbance_1', 0)
        # dv_1.set_writable()

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

        ## Define atributes
        # self.perturbations = [dv_1]
        self.inputs = [u_1, u_2]
        self.outputs = [y_1, y_2, y_3, y_4]

    def subscriptions(self):
        handler_inputs = SubHandler()
        handler_outputs = SubHandler()

        ## Define subscription update time (in ms)
        sub_inputs = self.server.create_subscription(100, handler_inputs)
        sub_outputs = self.server.create_subscription(100, handler_outputs)

        ## Subscriptions
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


if __name__ == '__main__':
    ## Start server
    server = OPC_server(plant_server_ip)
    server.new_namespace(uri='QuadrupleTank', namespace=QuadrupleTaknks_Namespace, name='QuadrupleTank_namespace')
    server.start()
