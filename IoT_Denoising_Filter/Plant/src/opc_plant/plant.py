import numpy as np

class NL_QuadrupleTank():
    def __init__(self, x0, Ts=0.1):
        self.x0 = x0
        self.error = 0

        # Parameters
        self.x = self.x0
        self.u = np.zeros((2,))
        self.Ts = Ts
        self.ti = 0
        self.current_time = 0
        self.voltmax = 10
        self.Hmax = 50

        # State-space system
        self.A = [28, 32, 28, 32] # cm^2
        self.a = [0.071, 0.057, 0.071, 0.057] # cm^2
        self.gamma = [0.7, 0.6] # %
        self.kin = 3.33
        self.g = 981 # cm/s^2

        # Variable auxiliar
        self.first_run = True 

        # Initial input (random)
        self.u = np.zeros((2,))

    def xd_func(self, u, x, t):
        # Ecuaciones diferenciales de los tanques
        xd0 = -self.a[0]/self.A[0]*np.sqrt(2*self.g*x[0]) + self.a[2]/self.A[0]*np.sqrt(2*self.g*x[2]) + self.gamma[0]*self.kin*u[0]*self.voltmax/self.A[0]
        xd1 = -self.a[1]/self.A[1]*np.sqrt(2*self.g*x[1]) + self.a[3]/self.A[1]*np.sqrt(2*self.g*x[3]) + self.gamma[1]*self.kin*u[1]*self.voltmax/self.A[1]
        xd2 = -self.a[2]/self.A[2]*np.sqrt(2*self.g*x[2]) + (1 - self.gamma[1])*self.kin*u[1]*self.voltmax/self.A[2]
        xd3 = -self.a[3]/self.A[3]*np.sqrt(2*self.g*x[3]) + (1 - self.gamma[0])*self.kin*u[0]*self.voltmax/self.A[3]
        res = [xd0, xd1, xd2, xd3]
        for i in range(len(res)):
            if np.isnan(res[i]) or type(res[i]) != np.float64:
                res[i] = 0
        return res
    
    def Limites(self):
        for i in range(len(self.x)):
            if self.x[i] > self.Hmax:
                self.x[i] = self.Hmax
            elif self.x[i] < 1e-2:
                self.x[i] = 1e-2

        for i in range(2):
            if self.u[i] > 1:
                self.u[i] = 1
            elif self.u[i] < -1:
                self.u[i] = -1

    def step(self, u):
        ## Input update
        self.u = u
        ## Time
        t = np.linspace(0, self.Ts, 2)
        ## "Discrete" update
        self.x = [x + self.Ts * y for x, y in zip(self.x, self.xd_func(self.u, self.x, t))]
        self.Limites()
        ## Increment time
        self.current_time += self.Ts

    

