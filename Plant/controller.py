class Controller():
    def __init__(self, Ts):
        self.Ts = Ts
        self.error = np.zeros((4,))
        self.u = np.zeros((2,))

        ## Define matrices
        # Use LQR to design the state feedback gain matrix K
        self.K = np.array([     [0.7844,    0.1129,   -0.0768,    0.5117],
                                [0.0557,    0.7388,    0.5409,   -0.0397]   ])
        # self.K = np.zeros((2,4))    # Zero input check (open loop)

        # Pole placement for tests
        # original poles x 0.5
        # self.K = np.array([     [0.0918,   -0.4558,    0.5289,    0.7562],
        #                         [-0.2061,  -0.0198,    0.6608,    0.4766]   ])
        # original poles x 0.1
        # self.K = np.array([     [-0.0136,   -0.0261,    0.0618,   -0.0933],
        #                         [ 0.0016,   -0.0230,   -0.1350,    0.0272]   ])
        # original poles x 1.1
        # self.K = np.array([     [0.7517,    0.1241,   -0.1805,    0.7416],
        #                         [0.1626,    1.0446,    0.4314,   -0.5294]   ])
        # original poles x 2
        # self.K = np.array([     [0.1629,    0.7565,   -1.1552,    4.8261],
        #                         [5.1158,    0.9324,    2.3980,   -14.1536]   ])
        # original poles x 10
        # self.K = np.array([     [-120.0466,  164.1560, -216.8447,  350.4219],
        #                         [108.0684,  -27.6837,   54.3643,  -291.0111]   ])

        # Integral controller
        # self.K_i = np.array([   [ 4.9097,   -0.0012,    0.1061,    0.0022,    1.0000,    0.0001],
        #                         [-0.0026,    5.6937,    0.0056,    0.1662,   -0.0001,    1.0000]   ]) # K by LQR with A* and B*
        # self.K_i = np.array([   [0.7844,    0.1129,   -0.0768,    0.5117,    1.0000,    0.0001],
        #                         [0.0557,    0.7388,    0.5409,   -0.0397,   -0.0001,    1.0000]   ]) # K with prior LQR and Identity
        # self.K_i = np.array([   [0.7517,    0.1241,   -0.1805,    0.7416,    1.0000,    0.0001],
        #                         [0.1626,    1.0446,    0.4314,   -0.5294,   -0.0001,    1.0000]   ]) # K * 1.1
        # self.K_i = np.array([     [0.0918,   -0.4558,    0.5289,    0.7562,    1.0000,    0.0001],
        #                           [-0.2061,  -0.0198,    0.6608,    0.4766,   -0.0001,    1.0000]   ]) # K * 0.5

        self.K_i = np.array([     [0.9107,   -0.0497,    0.1049,    0.0037,    0.0039,   -0.0004],
                                  [-0.0475,    1.4749,    0.0072,    0.1484,    0.0002,    0.0159]   ]) # K_1
        
        # self.K_i = np.array([       [-0.0219,    0.2058,   -0.2329,    0.2892,    0.0004,   -0.0004],
        #                             [0.1092,    0.2045,   -0.2201,   -0.2504,   -0.0001,    0.0002]   ]) # K_lim

    ## "Real time" integration
    def closed_loop(self, x_in, ref = np.zeros((4,))):
        # Calculate control input u = -Kx
        self.u = -self.K @ (x_in - ref)

    ## Open loop dynamics: input is u (2x1) and output is x (4x1)
    def open_loop(self, u):
        ## Input
        self.u = u

    ## Control with integral squeme
    def integral_control(self, x_in, ref = np.zeros((4,))):
        ## Error
        e = x_in - ref
        self.error += e * self.Ts
        ## Calculate control input u = -K_i(x + e)
        x_concat = np.hstack((x_in, self.error[:2])) # Error only for x_0 and x_1
        self.u = -self.K_i @ x_concat