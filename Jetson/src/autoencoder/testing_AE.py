import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
import json
import torch
import copy
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_metadata(meta, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + 'metadata.txt', 'w') as f:
        for key, value in meta.items():
            f.write('{}: {} \n'.format(key, value))

    with open(folder + 'meta.json', 'w') as f:
        json.dump(meta, f)

def load_metadata(folder):
    with open(folder + 'meta.json') as f:
        return json.load(f)


class TestingAutoencoder():
    def __init__(self, processor, model, data_type='Tanks', folder='/', path='model', batch_size=64, noise='white',
                 shuffle_train=True, missing_prob=0.2, noise_power=1, noise_inputs=False):
        self.processor = processor
        self.model = model
        self.noise_power = noise_power
        self.data_type = data_type
        self.noise = noise
        self.missing_prob = missing_prob
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.noise_inputs = noise_inputs

        self.criterion = nn.MSELoss(reduction='sum')
        self.initial_lr = 1e-3
        self.final_lr = 5e-5
        epochs = 250
        self.gamma = (1 / epochs) * np.log(self.final_lr / self.initial_lr)
        self.T = 30
        lambda_func = lambda epoch: np.exp(self.gamma * epoch)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        ## sheduler:    Sets the learning rate of each parameter group to the initial lr times a given function. When last_epoch=-1, 
        #               sets initial lr as lr. En nuestros ejemplos, no se usará
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_func)


        self.patience = 30
        self.no_better = 0
        self.train_losses = []
        self.eval_losses = []
        self.training_times = []
        self.inference_times = []

        ## Create folder
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.path = path
        self.path_best = 'best_' + path
        self.min_eval_loss = 1000000000
        # Statistics
        self.h_statistics = {'pca_error': [], 'total_variance': [], 'cosine_sim': [], 'mean_variance': [],
                             'pca_error_add': [], 'total_variance_add': [], 'cosine_sim_add': [],
                             'mean_variance_add': []}

    ## Función estrictamente creciente con las iteraciones que satura en 1
    def increaseMissingProb(self, it):
        periodDecay = 75
        missingProb = min(1, 0.2 + it * (0.2 / periodDecay))
        return missingProb

    def generate_batch(self, X, p=None, batch_size=64, seqlen=60, mode='random'):
        size = X.shape[0]
        if mode == 'random':
            batch = []
            random_numbers = [random.randint(0, size - 2) for i in range(int(batch_size/2))] # numeros aleatorios
            for r in random_numbers:
                batch.append(X[r: r+2, :, :])
            return torch.cat(batch, dim=0)

        elif mode == 'seq':
            return X[p: p + batch_size, :, :]

        else: # mode == 'combined'
            batch = []
            ## 1. Genero lista con los elementos del batch (los que van si o si)
            batch.append(X[p: p + batch_size, :, :])
            ## 2. Genero todos los demás índice de datos que no están en el batch actual
            numbers = [i for i in range(0, p-1)] + [i for i in range(p+batch_size, size)]
            ## 3. De forma random, escojo 'batch_size' índices de los que NO estaban en el batch actual
            random_numbers = random.choices(numbers, k=batch_size)
            ## 4. Para cada índice random escogido, se appendea al batch "real"
            for r in random_numbers:
                batch.append(X[r, :, :].reshape(1, seqlen, -1))
            ## Resulta: un batch de 2*batch_size, seqlen, n_inputs
            return torch.cat(batch, dim=0)

    ## Función comentada por el momento
    def track_statistics(self, h_list, h_add_list):
        h = np.concatenate(h_list, axis=0)
        layer_len = int(h.shape[1]/2)
        h = h[:, -layer_len:]

        h_add = np.concatenate(h_add_list, axis=0)

        # PCA
        pca_errors = []
        pca_add_errors = []
        for i in range(10, 16):
            pca = PCA(n_components=i)
            pca.fit(h)
            h_pca = pca.inverse_transform(pca.transform(h))
            pca_errors.append(root_mean_squared_error(h_pca, h))

            pca_add = PCA(n_components=i)
            pca_add.fit(h_add)
            h_pca_add = pca_add.inverse_transform(pca_add.transform(h_add))
            pca_add_errors.append(root_mean_squared_error(h_pca_add, h_add))
        self.h_statistics['pca_error'].append(pca_errors)
        self.h_statistics['pca_error_add'].append(pca_add_errors)

        # Mean and Variance
        self.h_statistics['total_variance'].append(h.var())
        self.h_statistics['mean_variance'].append(np.mean(h.var(axis=0)))

        self.h_statistics['total_variance_add'].append(h_add.var())
        self.h_statistics['mean_variance_add'].append(np.mean(h_add.var(axis=0)))

        # Cosine Similarity
        cosine_sum = 0
        cosine_sum_add = 0
        for i in range(h.shape[0] - 1):
            cosine_sum += cosine_similarity(h[i, :].reshape(1, -1), h[i + 1, :].reshape(1, -1))[0, 0]
            cosine_sum_add += cosine_similarity(h_add[i, :].reshape(1, -1), h_add[i + 1, :].reshape(1, -1))[0, 0]

        self.h_statistics['cosine_sim'].append(cosine_sum / (h.shape[0] - 1))
        self.h_statistics['cosine_sim_add'].append(cosine_sum_add / (h_add.shape[0] - 1))

        torch.save(self.h_statistics, self.folder + 'h_statistics.pkl')      

    def eval(self, missing=False, n_inputs=8, plot=False, name=''):  # Checkpoints deben ir ordenados de mayor a menor

        if self.data_type == 'Tanks':
            ## data_dict tiene shuffle
            self.data_dict = self.processor.process_tanks(noise_power=self.noise_power, noise_inputs=self.noise_inputs, shuffle=self.shuffle_train, 
                                                          folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=self.noise, clean_data='data_NL_clean.pkl')
            ## data_dict2 no tiene shuffle
            self.data_dict2 = self.processor.process_tanks(noise_power=self.noise_power, noise_inputs=self.noise_inputs, shuffle=False,
                                                           folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=self.noise, clean_data='data_NL_clean.pkl')
        else:
            self.data_dict = self.processor.process_tickener(shuffle=self.shuffle_train)
            self.data_dict2 = self.processor.process_tickener(shuffle=False)

        self.data_train = self.data_dict['train_data']
        self.data_test = self.data_dict2['test_data']
        self.scaler = self.data_dict['scaler']
        self.scaler_preproc = self.data_dict['scaler_preproc']

        ## Escogemos la porción de datos de testing para hacer la evaluación: test_seq tiene los datos con ruido, clean_seq los originales sin ruido
        test_seq = self.data_dict2['test_data']
        clean_seq = self.data_dict2['test_data_preproc']

        ### Evaluación
        length_eval = self.data_test.shape[0]
        ## Inicia modo de evaluación: model.eval() is a kind of switch for some specific layers/parts of the model that behave differently 
        ## during training and inference (evaluating) time (https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
        self.model.eval() 
        results_dict = {}
        print('\n Evaluating {}'.format(name))
        ## Por cada entrada, recorro una vez el for
        
        missing_index = 2
        print('\n Analizing missing index: {}'.format(missing_index + 1))
        p = 0
        X_in_list = []
        X_in_no_missing_list = []
        Y_pred_list = []
        Y_pred_no_missing_list = []
        h_seq_list = []
        h_seq_no_missing_list = []
        X_clean_list = []
        missing_index_dict = {}
        with torch.no_grad():
            ## length_eval es el número total de columnas de data en testeo
            while p < length_eval:
                ## X_seq es el batch analizado, que va desde la columna p hasta p+batch_size y contiene datos con ruido
                X_seq = test_seq[p:p + self.batch_size, :, :].to(device)
                ## X_in = X_seq
                X_in = copy.deepcopy(X_seq)

                ## X_clean es el batch analizado, que va desde la columna p hasta p+batch_size
                X_clean = clean_seq[p:p + self.batch_size, :, :]
                ## Actualizamos el batch count (puntero)
                p += self.batch_size

                ### Predicciones: detalle en línea 239
                Ypred, _, h_seq = self.model(X_in)
                Ypred_no_missing, _, h_seq_no_missing = self.model(X_seq)

                # Listas
                X_in_list.append(X_in[:, -1, :].cpu().numpy())
                X_in_no_missing_list.append(X_seq[:, -1, :].cpu().numpy())
                Y_pred_list.append(Ypred[:, -1, :].cpu().numpy())
                Y_pred_no_missing_list.append(Ypred_no_missing[:, -1, :].cpu().numpy())
                h_seq_list.append(h_seq[1:, :, :].cpu().numpy().reshape(Ypred.shape[0], -1))
                h_seq_no_missing_list.append(h_seq_no_missing[1:, :, :].cpu().numpy().reshape(Ypred.shape[0], -1))
                X_clean_list.append(X_clean[:, -1, :].cpu().numpy())

        # Concatenación y escalamiento de los resultados
        X_in_list = self.scaler.inverse_transform(np.concatenate(X_in_list, axis=0))
        X_in_no_missing_list = self.scaler.inverse_transform(np.concatenate(X_in_no_missing_list, axis=0))
        Y_pred_list = self.scaler.inverse_transform(np.concatenate(Y_pred_list, axis=0))
        Y_pred_no_missing_list = self.scaler.inverse_transform(np.concatenate(Y_pred_no_missing_list, axis=0))
        X_clean_list = self.scaler_preproc.inverse_transform(np.concatenate(X_clean_list, axis=0))
        h_seq_list = np.concatenate(h_seq_list, axis=0)
        h_seq_no_missing_list = np.concatenate(h_seq_no_missing_list, axis=0)

        # for i in range(n_inputs):
        #     plt.figure(i)
        #     plt.plot(Y_pred_list[:, i], label='Y_AE')
        #     plt.plot(X_clean_list[:, i], label='Y_clean')
        #     plt.legend()
        #     plt.title(f'Indice {i}')
        # plt.show()

        ## All inputs and outputs
        # plt.figure(7)
        # for i in range(n_inputs):
        #     plt.subplot(int(f'32{i+1}'))
        #     plt.plot(X_in_list[:, i], label=f'Y_noisy_{i}')
        #     plt.plot(Y_pred_list[:, i], label=f'Y_AE_{i}')
        #     plt.plot(X_clean_list[:, i], label=f'Y_clean_{i}')
        #     plt.legend()
        #     plt.title(f'Indice {i}')
        # plt.show()

        ## Just outputs
        plt.figure(8)
        for i in range(4):
            plt.subplot(int(f'22{i+1}'))
            plt.plot(X_in_list[:, i+2], label=f'Y_noisy_{i}')
            plt.plot(Y_pred_list[:, i+2], label=f'Y_AE_{i}')
            plt.plot(X_clean_list[:, i+2], label=f'Y_clean_{i}')
            plt.legend()
            plt.title(f'x_{i}')
        plt.show()

        # Performances calculation
        total_rmse = root_mean_squared_error(Y_pred_list[:, 2:], X_clean_list[:, 2:])
        lost_index_rmse = root_mean_squared_error(Y_pred_list[:, missing_index], X_clean_list[:, missing_index])

        print('Total RMSE: {}'.format(total_rmse))
        # print('Lost index RMSE: {}'.format(lost_index_rmse))

        total_rmse_no_missing = root_mean_squared_error(Y_pred_no_missing_list[:, 2:], X_clean_list[:, 2:])
        lost_index_rmse_no_missing = root_mean_squared_error(Y_pred_no_missing_list[:, missing_index], X_clean_list[:, missing_index])
        # print('Total RMSE no missing: {}'.format(total_rmse_no_missing))
        # print('Lost index RMSE no missing: {}'.format(lost_index_rmse_no_missing))


        # Cosine Similarity
        cosine_sum = 0
        for i in range(h_seq_list.shape[0]):
            cosine_sum += cosine_similarity(h_seq_list[i, :].reshape(1, -1), h_seq_no_missing_list[i, :].reshape(1, -1))[0, 0]
        cosine_sum = cosine_sum/h_seq_list.shape[0]

        print('Average Cosine Similarity {}'.format(cosine_sum))

        missing_index_dict['total_rmse'] = total_rmse
        missing_index_dict['lost_index_rmse'] = lost_index_rmse
        missing_index_dict['total_rmse_no_missing'] = total_rmse_no_missing
        missing_index_dict['lost_index_rmse_no_missing'] = lost_index_rmse_no_missing
        missing_index_dict['cosine_similarity'] = cosine_sum
        missing_index_dict['X_in_no_missing'] = X_in_no_missing_list[:, missing_index]
        missing_index_dict['Y_pred'] = Y_pred_list[:, missing_index]
        missing_index_dict['Y_pred_no_missing'] = Y_pred_no_missing_list[:, missing_index]
        missing_index_dict['X_clean'] = X_clean_list[:, missing_index]
        results_dict['{}'.format(missing_index + 1)] = missing_index_dict

        torch.save(results_dict, self.folder + 'reconstruction_results.pkl')

    