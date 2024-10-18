import numpy as np
from params import *
from src.plant.NL_QuadrupleTank import NL_QuadrupleTank
from scipy.integrate import odeint

class new_NL_QuadrupleTank():
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

    def xd_func(self, x, t):
        # Ecuaciones diferenciales de los tanques
        xd0 = -self.a[0]/self.A[0]*np.sqrt(2*self.g*x[0]) + self.a[2]/self.A[0]*np.sqrt(2*self.g*x[2]) + self.gamma[0]*self.kin*self.u[0]*self.voltmax/self.A[0]
        xd1 = -self.a[1]/self.A[1]*np.sqrt(2*self.g*x[1]) + self.a[3]/self.A[1]*np.sqrt(2*self.g*x[3]) + self.gamma[1]*self.kin*self.u[1]*self.voltmax/self.A[1]
        xd2 = -self.a[2]/self.A[2]*np.sqrt(2*self.g*x[2]) + (1 - self.gamma[1])*self.kin*self.u[1]*self.voltmax/self.A[2]
        xd3 = -self.a[3]/self.A[3]*np.sqrt(2*self.g*x[3]) + (1 - self.gamma[0])*self.kin*self.u[0]*self.voltmax/self.A[3]
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

    ## Update state using A - BK dynamic
    def step(self, u):
        ## Input update
        self.u = u
        
        ## Time
        t = np.linspace(0, self.Ts, 2)

        ## "Real time" update
        x = odeint(self.xd_func, self.x, t)  # Perform integration using Fortran's LSODA (Adams & BDF methods)
        self.x = [x[-1, 0], x[-1,1], x[-1, 2], x[-1, 3]]

        ## "Discrete" update
        # self.x = [x + self.Ts * y for x, y in zip(self.x, self.xd_func(self.u, self.x, t))]
        self.Limites()

        # Increment time
        self.current_time += self.Ts

if __name__ == '__main__':
    ## Instanciate Plant
    x0=[12.4, 12.7, 1.8, 1.4]
    plant = NL_QuadrupleTank(x0=x0, Ts=0.001)
    new_plant = new_NL_QuadrupleTank(x0=x0, Ts=0.001)

    cnt = 0
    x_hist, x_new_hist, u_hist = [], [], []
    while True:
        ## Get the values from the controller and update the plant
        
        ## Plant step
        u_aux = np.array([np.random.uniform(-3,5), np.random.uniform(-3,5)])
        u_aux = np.array([1,-1])
        f = 1
        a = 1
        u_aux = [a*np.sin(cnt*f), a*np.cos(cnt*f)]
        plant.step(u_aux)
        new_plant.step(u_aux)

        # print(u_aux)
        # print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(plant.x[0], plant.x[1], plant.x[2], plant.x[3]))
        # print('(y_1, y_2, y_3, y_4) = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(new_plant.x[0], new_plant.x[1], new_plant.x[2], new_plant.x[3]))
        # print(50*'-')
        u_hist += [u_aux]
        x_hist += [plant.x]
        x_new_hist += [new_plant.x]
        # Add a delay if needed
        # time.sleep(1)
        cnt += 1

        if cnt == 10000:
            break

    x_hist = np.array(x_hist)
    x_new_hist = np.array(x_new_hist)
    u_hist = np.array(u_hist)

    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    plt.plot(x_hist[:,0], label='x_1')
    plt.plot(x_new_hist[:,0], label='x_new_1')
    plt.legend()
    plt.grid()
    plt.subplot(2,2,2)
    plt.plot(x_hist[:,1], label='x_2')
    plt.plot(x_new_hist[:,1], label='x_new_2')
    plt.legend()
    plt.grid()
    plt.subplot(2,2,3)
    plt.plot(x_hist[:,2], label='x_3')
    plt.plot(x_new_hist[:,2], label='x_new_3')
    plt.legend()
    plt.grid()
    plt.subplot(2,2,4)
    plt.plot(x_hist[:,3], label='x_4')
    plt.plot(x_new_hist[:,3], label='x_new_4')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(u_hist[:,0], label='u_1')
    plt.plot(u_hist[:,1], label='u_2')
    plt.legend()
    plt.grid()
    plt.show()