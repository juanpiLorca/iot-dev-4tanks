import numpy as np
import torch

## Cargar datos LF y HF
lf_data = np.load('No_Noised_Inputs/LF_data_NL_clean.npy')
hf_data = np.load('No_Noised_Inputs/HF_data_NL_clean.npy')
print(f'Shapes: LF = {lf_data.shape} -- HF = {hf_data.shape}')

## Unir ambos
data = np.concatenate((hf_data, lf_data))
print(f'Shape data = {data.shape}')

## Guardar data
np.save('No_Noised_Inputs/data_NL_clean.npy', data)
torch.save(data, 'No_Noised_Inputs/data_NL_clean.pkl')
print(f'Guardado completado!')