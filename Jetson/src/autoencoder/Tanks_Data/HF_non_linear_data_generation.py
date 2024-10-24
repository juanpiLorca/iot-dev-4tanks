import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
import random
import threading
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pandas as pd
from tqdm import tqdm
import torch, os
import sys
sys.path.append('../')
from non_linear_QuadrupleTank_class import NL_QuadrupleTank

def generate_pattern(n_seqs=500, seqlen=60, window=20):
    u1_seqs = []
    u2_seqs = []
    ## Semilla en caso de ser necesaria
    random.seed(321)
    ## Generar vector con n_seq secuencias de largo seqlen, en que cada una contiene un número aleatorio >= 0.05
    for n in range(n_seqs):
        u1_val = random.uniform(-1, 1)
        u2_val = random.uniform(-1, 1)

        u1_seqs.append([u1_val for i in range(seqlen)])
        u2_seqs.append([u2_val for i in range(seqlen)])

    ## Agregar muestras que se pierden en el filtrado Hamming
    u1_val = random.uniform(-1, 1)
    u2_val = random.uniform(-1, 1)
    u1_seqs.append([u1_val for i in range(window-1)])
    u2_seqs.append([u2_val for i in range(window-1)])

    ## Estirar secuencias para que quede una lista de largo n_seq * seqlen
    u1_seqs = np.concatenate(u1_seqs).reshape(-1, 1)
    u2_seqs = np.concatenate(u2_seqs).reshape(-1, 1)

    ## Matriz de secuencias: n_seq * seqlen x 2
    u_seqs = np.concatenate([u1_seqs, u2_seqs], axis=1)
    ## Sumar ruido gaussiano con media = 0 y std = 0.01
    u_seqs = u_seqs + np.random.normal(loc=np.zeros_like(u_seqs), scale=np.ones_like(u_seqs)*0.01)
    ## Hace una ventana de tamaño window y la aplica la función mean(). Los primeros 19 valores son NaN, luego empieza la media
    u_seqs = np.array(pd.DataFrame(u_seqs).rolling(window=window, min_periods=0, win_type='hamming').mean())
    ## Sumar ruido gaussiano con media = 0 y std = 0.0015
    u_seqs = u_seqs + np.random.normal(loc=np.zeros_like(u_seqs), scale=np.ones_like(u_seqs)*0.0015)
    ## Comentado de antes
    #u_seqs = savgol_filter(u_seqs, window_length=21, polyorder=3, axis=0)
    
    ## Retornar la secuencia sin los Nan
    return u_seqs[(window-1):,:]

if __name__ == '__main__':
    
    ones = [1,1,1,1]
    x0 = [x * 30 for x in ones]
    print(f'x0 = {x0}')
    sistema = NL_QuadrupleTank(x0=x0)
    sistema.time_scaling = 1 # Para el tiempo

    ## Mostrar hasta esa muestra
    upto = 4000

    ## Input generation
    inputs = generate_pattern(n_seqs=5000, seqlen=10, window=3) 
    inputs *= 10
    print(f'inputs = {inputs.shape}')

    # plt.figure(1)
    # for i in range(2):
    #     plt.plot(inputs[:upto, i], label=f'u_{i}')
    # plt.title('Inputs')
    # plt.legend()
    # plt.grid()
    # plt.show()

    ## Sampleo continuo cada Ts_c segundos
    Ts_c = 0.1
    aux = int(1 / Ts_c)

    series = np.zeros((len(inputs), 4))
    continuo = np.zeros((aux * len(inputs), 4))

    cnt = 0
    for i in tqdm(range(len(inputs))):
        u_input = list(inputs[i,:])
        # print(inputs[i,:])
        sistema.open_loop(u=u_input, Ts_c=Ts_c)
        # print(sistema.x, sistema.u)
        # series.append(sistema.x)
        # continuo.append(sistema.x)
        series[i, :] = sistema.x
        continuo[cnt, :] = sistema.x
        cnt += 1
        for _ in range(aux - 1):
            sistema.open_loop(u=u_input, Ts_c=Ts_c)
            # continuo.append(sistema.x)
            continuo[cnt] = sistema.x
            cnt += 1
            
    series = np.array(series)
    continuo = np.array(continuo)
    tiempo_total = np.arange(0, len(inputs), Ts_c)
    tiempo_sample = np.arange(0, len(inputs), 1)
    print(f'tiempo continuo = {tiempo_total.shape}')
    print(f'tiempo sampleo = {tiempo_sample.shape}')
    print(f'series.shape = {series.shape}')
    print(f'continuo.shape = {continuo.shape}')

    ## Truncate the input series to eliminate the NaN values
    # inputs = inputs[-series.shape[0]:, :]
    # print(f'inputs.shape = {inputs.shape}')

    upto = 2*10**3
    # plt.figure(2)
    # plt.title('Alturas de los tanques [cm] sampleado')
    # plt.plot(tiempo_sample[:upto], series[:upto, 0], label='Tanque1')
    # plt.plot(tiempo_sample[:upto], series[:upto, 1], label='Tanque2')
    # plt.plot(tiempo_sample[:upto], series[:upto, 2], label='Tanque3')
    # plt.plot(tiempo_sample[:upto], series[:upto, 3], label='Tanque4')
    # plt.legend()
    # plt.grid()
    

    # plt.figure(3)
    # plt.title('Inputs')
    # plt.plot(tiempo_sample[:upto], inputs[:upto, 0], label='Input1')
    # plt.plot(tiempo_sample[:upto], inputs[:upto, 1], label='Input2')
    # plt.grid()
    # plt.legend()

    # plt.figure(4)
    # plt.title('Alturas de los tanques [cm] continuo')
    # plt.plot(tiempo_total, continuo[:, 0], label='Tanque1')
    # plt.scatter(tiempo_sample, series[:, 0], label='Discreto', marker='*', color='red')
    # plt.plot(tiempo_total, continuo[:, 1], label='Tanque2')
    # plt.plot(tiempo_total, continuo[:, 2], label='Tanque3')
    # plt.plot(tiempo_total, continuo[:, 3], label='Tanque4')
    # plt.legend()
    # plt.grid()

    # Estanque 1 y 2 en continuo y discreto
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(tiempo_total, continuo[:, 0], color='blue')
    # axs[1].plot(tiempo_total, continuo[:, 1], color='red')
    # axs[0].scatter(tiempo_sample, series[:, 0], color='green', marker='*')
    # axs[1].scatter(tiempo_sample, series[:, 1], color='purple', marker='*')
    # for i in range(2):
    #     axs[i].set_title(f'Tanque {i+1}')
    #     axs[i].grid()
    #     axs[i].set_ylim(-5, 55)
    # plt.tight_layout()

    # Todos los estanques
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(series[:, 0], color='blue')
    axs[1].plot(series[:, 1], color='red')
    axs[2].plot(series[:, 2], color='green')
    axs[3].plot(series[:, 3], color='purple')
    for i in range(4):
        axs[i].set_title(f'Tanque {i+1}')
        axs[i].grid()
        axs[i].set_ylim(-5, 55)
    plt.tight_layout()

    # Estanque 1 y 2 en discreto
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(series[:, 0], color='blue')
    axs[1].plot(series[:, 1], color='red')
    for i in range(2):
        axs[i].set_title(f'Tanque {i+1}')
        axs[i].grid()
        axs[i].set_ylim(-5, 55)
    plt.tight_layout()
    plt.show()

    # plt.figure(3)
    # plt.title('Estados generados')
    # plt.plot(series[:, 0], label='x_0')
    # plt.plot(series[:, 1], label='x_1')
    # plt.plot(series[:, 2], label='x_2')
    # plt.plot(series[:, 3], label='x_3')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    ## Crear directorio
    if not os.path.exists('No_Noised_Inputs/'):
            os.makedirs('No_Noised_Inputs/')

    ## Guardar data
    data_to_save = np.concatenate([inputs, series], axis=1)
    print(data_to_save.shape)

    ## Chequeo de NaN values
    has_nan = np.isnan(np.array(data_to_save)).any()
    if has_nan:
        print("The matrix contains NaN values.")
        data_to_save = np.array([[0 if np.isnan(element) else element for element in row] for row in data_to_save])

    has_nan = np.isnan(np.array(data_to_save)).any()
    if has_nan:
        print("The matrix STILL contains NaN values.")
    else:
        print("NaN values were deleted.")
    print(data_to_save.shape)
    np.save('No_Noised_Inputs/HF_data_NL_clean.npy', data_to_save)
    # np.save('data_noise.npy', np.concatenate([inputs_noise, series_noise], axis=1))
    torch.save(data_to_save, 'No_Noised_Inputs/HF_data_NL_clean.pkl')
    print(f'Saved \'data_to_save\'')
    

