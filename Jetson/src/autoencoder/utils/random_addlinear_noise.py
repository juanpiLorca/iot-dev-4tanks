import os
import copy
import torch
import time
import json
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.nets import RNNAutoencoder_AddLayer
from utils.nce_loss import random_seq_contrastive_loss
from utils.data_processor import DataProcessor
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


class TrainingAutoencoder():
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
        
        if self.data_type == 'Tanks':
            ## data_dict tiene shuffle
            self.data_dict = self.processor.process_tanks(noise_power=noise_power, noise_inputs=noise_inputs, shuffle=shuffle_train, 
                                                          folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=self.noise, clean_data='data_NL_clean.pkl')
            ## data_dict2 no tiene shuffle
            self.data_dict2 = self.processor.process_tanks(noise_power=noise_power, noise_inputs=noise_inputs, shuffle=False,
                                                           folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=self.noise, clean_data='data_NL_clean.pkl')

        else:
            self.data_dict = self.processor.process_tickener(shuffle=shuffle_train)
            self.data_dict2 = self.processor.process_tickener(shuffle=False)

        self.data_train = self.data_dict['train_data']
        self.data_test = self.data_dict2['test_data']
        self.scaler = self.data_dict['scaler']
        self.scaler_preproc = self.data_dict['scaler_preproc']

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
        
        # datos de tanques de training y validación
        train_seq = self.data_dict2['train_data']
        val_seq = self.data_dict2['val_data']
        clean_seq = self.data_dict2['val_data_preproc']
        
        ## Chequeo de sets correctos para posterior training
        # print(f'train_seq = {train_seq.shape}')
        # print(f'val_seq = {val_seq.shape}')
        # print(f'clean_seq = {clean_seq.shape}')

        # for i in range(6):
        #     plt.figure(i)
        #     plt.plot(val_seq[:300,-1,i], label='val_seq')
        #     plt.plot(clean_seq[:300,-1,i], label='clean_seq')
        #     plt.legend()
        #     plt.title(f'idx = {i}')
        # plt.show()

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


    def train(self, epochs=300, checkpoints=[], missing=False, scheduler=False, n_inputs=8, criterion='MSE',
              beta=1, tau=1, seqlen=60, mode='random'): # Checkpoints deben ir ordenados de mayor a menor
        
        ### Función de error escogida
        if criterion == 'DTW':
            self.criterion = nn.MSELoss() #SoftDTW(gamma=0.5, normalize=True)
        elif criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.SmoothL1Loss()

        # data_train = shape : , seqlen, n_inputs
        length = self.data_train.shape[0]
        eval_responses = []
        eval_responses_no_loss = []

        # datos de tanques de training y validación
        train_seq = self.data_dict2['train_data']
        val_seq = self.data_dict2['val_data']
        clean_seq = self.data_dict2['val_data_preproc']

        for epoch in range(epochs):
            losses = 0
            contrastive_losses = 0
            p = 0
            its = 0
            self.model.train()
            init_time = time.time()
            X_in_test_list = []
            X_out_test_list = []

            # Mientras no se acaben los datos 
            while p < length:
                its += 1
                # Cada 50 batches, printea
                if p % (50*self.batch_size) == 0 and p!=0:
                    print('epoch:{} | its: {}:{} | Train loss: {}'.format(epoch, length, p, losses/p))
                    self.train_losses.append([losses/p, contrastive_losses/p])
                    torch.save(self.train_losses, self.folder + 'train_losses.pkl')

                # Desacrivamos actualización de gradientes
                self.optimizer.zero_grad()
                # Generamos un batch de data, de shape 2*batch_size, seqlen, n_inputs
                X_seq = self.generate_batch(train_seq, p=p, batch_size=self.batch_size, seqlen=seqlen, mode=mode).to(device) #train_seq[p:p + self.batch_size, :, :].to(device)

                # X_in es una copia de X_seq
                X_in = copy.deepcopy(X_seq)
                
                # Simulamos que un sensor tira una medición errónea
                if missing:
                    # Función estrictamente creciente con las iteraciones que satura en 1 -> a medida que itero más, 'missing_prob' será
                    # cada vez más cerca de 1
                    missing_prob = self.increaseMissingProb(epoch)
                    # Con mayores iteraciones, 'miss_sensor' es cada vez más frecuentemente True
                    miss_sensor = True if random.random() < missing_prob else False
                    if miss_sensor:
                        # Se escoje un sensor y se reemplaza la altura por 10 (error de medición)
                        miss_index = random.choice([k for k in range(2, n_inputs)])
                        X_in[:, :, miss_index] = torch.ones_like(X_in[:, :, miss_index]).to(device) * (10)

                # Actualizamos valor del batch index
                p += self.batch_size

                ### Gradients
                # Valores de la predicción Ypred, de h_seq y h respectivamente
                #       h_seq:  contains a sequence of hidden states or representations generated by the RNN at each time step as it processes 
                #               the input sequence. Each element of h_seq corresponds to the hidden state of the RNN at a specific time step. 
                #               This sequence can be valuable for tasks where you want to analyze how the RNN's internal representations evolve over time.
                #       h:      represents the final hidden state of the RNN after processing the entire input sequence X_in. It's a summary or aggregation 
                #               of all the hidden states in h_seq. This final hidden state is often used as a context or summary of the entire input sequence
                #               and can be useful for tasks like sequence-to-sequence modeling or when you need to condense the information from the entire 
                #               sequence into a fixed-size representation.
                Ypred, h_seq, h = self.model(X_in) # shapes: h_seq = 2*batch_size, n_layers ; h = n_layers, 2*batch_size, hidden_size
                
                # Cálculo de Noise Contrastive Loss
                contrastive_loss = random_seq_contrastive_loss(h=h_seq.unsqueeze(0), tau=tau, mode=mode)

                ## Pérdida: 𝐿 = 𝐿𝐴𝐸 + 𝛽 𝐿𝑁𝐶𝐸
                loss = torch.mean(self.criterion(X_seq, Ypred)) + beta*contrastive_loss
                # Backward pass
                loss.backward()
                losses += float(loss.data)
                contrastive_losses += float(contrastive_loss.data)
                # Step
                self.optimizer.step()
            ### Fin: se terminaron los batches de datos ###

            # Tiempo que tomó en correr UNA época
            total_time = time.time() - init_time

            # Tiempo de entrenamiento y guardado
            self.training_times.append(total_time)
            torch.save(self.training_times, self.folder + 'trainig_times.pkl')
            # Parámetros de la red y guardado
            #   -> model.state_dict(): returns the learnable parameters (i.e. weights and biases) of a torch.nn.Module model
            torch.save(self.model.state_dict(), self.folder + self.path)

            # Función decreciente que satura en cero -> sirve para hacer el switch entre entrenamiento con muestras pasadas y actuales
            self.model.decay_teacher(epoch)

            ## Setea el learning rate, descrito más arriba (seteado en False en nuestros ejemplos)
            if scheduler:
                self.scheduler.step()

            ### Evaluación
            p = 0
            length_eval = self.data_test.shape[0]
            losses = 0
            contrastive_losses = 0
            self.model.eval()
            # Tiempo que demora el modelo en dar predicciones
            total_times = []
            pred_eval = []
            pred_eval_no_loss = []
            real_eval = []
            h_test_list = []
            h_add_test_list = []
            X_in_test_list = []
            X_out_test_list = []
            X_clean_test_list = []

            ## Evaluación sin actualizar los gradientes: la cuenta se reinicia (p = 0)
            with torch.no_grad():
                while p < length_eval:
                    its += 1
                    ## Nuestras secuencias, repartidas en batches, son ahora las de validación (val_seq)
                    #  Seguimos exactamente el mismo procedimiento del training
                    X_seq = val_seq[p:p + self.batch_size, :, :].to(device)
                    # X_in = X_seq
                    X_in = copy.deepcopy(X_seq)
                    # Se escoge un sensor "malo" al azar y se reemplaza su valor (simulando falla en la medición)
                    if missing:
                        miss_sensor = True
                        if miss_sensor:
                            miss_index = random.choice([k for k in range(2, n_inputs)])
                            X_in[:, :, miss_index] = torch.ones_like(X_in[:, :, miss_index]).to(device) * (10)
                    # Batch de secuencia sin ruido
                    X_clean = clean_seq[p:p + self.batch_size, :, :]

                    # Actualizar pocisión del batch
                    p += self.batch_size

                    init_time = time.time()

                    ### Gradients
                    # Aquí las dimensiones son: h_seq = batch_size, n_layers ; h = n_layers, batch_size, hidden_size
                    Ypred, h_seq, h = self.model(X_in)
                    Ypred_no_loss, _, _ = self.model(X_seq)

                    # total_times: contiene el cálculo del tiempo que demora el modelo en entregar las predicciones
                    total_times.append(time.time() - init_time)
                    # X_in_test_list: contiene cada batch de data con ruido 
                    X_in_test_list.append(X_seq.detach().cpu().numpy())
                    # X_out_test_list: contiene la predicción del modelo para cada batch
                    X_out_test_list.append(Ypred.detach().cpu().numpy())
                    # X_clean_test_list: contiene cada batch de data sin ruido 
                    X_clean_test_list.append(X_clean.detach().cpu().numpy())
                    # 
                    h_test_list.append(h.detach().cpu().numpy().reshape(Ypred.shape[0], -1))
                    #
                    h_add_test_list.append(h_seq.detach().cpu().numpy().reshape(Ypred.shape[0], -1))

                    pred_eval.append(Ypred[:, -1, :].cpu().numpy())
                    pred_eval_no_loss.append(Ypred_no_loss[:, -1, :].cpu().numpy())
                    real_eval.append(X_seq[:, -1, :].cpu().numpy())
                    contrastive_loss = random_seq_contrastive_loss(h=h_seq, mode=mode)
                    loss = torch.mean(self.criterion(X_seq, Ypred)) + beta * contrastive_loss
                    losses += float(loss.data)
                    contrastive_losses += float(contrastive_loss.data)

            
            print('Eval Loss: {}| {}'.format(losses / p, contrastive_losses / p))
            self.inference_times.append(np.mean(np.array(total_times)))
            eval_responses.append(np.concatenate(pred_eval, axis=0))
            eval_responses_no_loss.append(np.concatenate(pred_eval_no_loss, axis=0))
            real_eval = np.concatenate(real_eval, axis=0)
            self.eval_losses.append([losses / p, contrastive_losses / p])
            #self.track_statistics(h_test_list, h_add_test_list)

            torch.save(self.inference_times, self.folder + 'inference_times.pkl')
            torch.save(self.eval_losses, self.folder + 'eval_losses.pkl')
            torch.save(eval_responses, self.folder + 'eval_response.pkl')
            torch.save(eval_responses_no_loss, self.folder + 'eval_response_no_loss.pkl')
            torch.save(real_eval, self.folder + 'real_response.pkl')
                        
            if losses / p < self.min_eval_loss:
                self.min_eval_loss = losses / p
                print('Guardando mejor modelo')
                self.no_better = 0
                torch.save([X_in_test_list, h_test_list, X_out_test_list, X_clean_test_list],
                           self.folder + 'h_test_list.pkl')
                torch.save(self.model.state_dict(), self.folder + self.path_best)

            else:
                self.no_better += 1
                if self.no_better >= self.patience:
                    print('Terminado Early Stopping')
                    return None
            

    def eval(self, missing=False, n_inputs=8, plot=False, name=''):  # Checkpoints deben ir ordenados de mayor a menor

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
        for missing_index in range(n_inputs):
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

                    ## missing lo que hace es: reemplazar la fila que se está viendo por un tensor de unos, en este caso multiplicado
                    ## por 10. Con esto se simula que el batch está lleno de una entrada mala
                    if missing:
                        ## Si missing = True, relleno el tensor con 10 en el 'missing_index'
                        X_in[:, :, missing_index] = torch.ones_like(X_in[:, :, missing_index]).to(device) * (10)

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

            # Performances calculation
            total_rmse = root_mean_squared_error(Y_pred_list[:, 2:], X_clean_list[:, 2:])
            lost_index_rmse = root_mean_squared_error(Y_pred_list[:, missing_index], X_clean_list[:, missing_index])

            print('Total RMSE: {}'.format(total_rmse))
            print('Lost index RMSE: {}'.format(lost_index_rmse))

            total_rmse_no_missing = root_mean_squared_error(Y_pred_no_missing_list[:, 2:], X_clean_list[:, 2:])
            lost_index_rmse_no_missing = root_mean_squared_error(Y_pred_no_missing_list[:, missing_index], X_clean_list[:, missing_index])

            print('Total RMSE no missing: {}'.format(total_rmse_no_missing))
            print('Lost index RMSE no missing: {}'.format(lost_index_rmse_no_missing))


            # Cosine Similarity
            cosine_sum = 0
            for i in range(h_seq_list.shape[0]):
                cosine_sum += cosine_similarity(h_seq_list[i, :].reshape(1, -1), h_seq_no_missing_list[i, :].reshape(1, -1))[0, 0]
            cosine_sum = cosine_sum/h_seq_list.shape[0]

            print('Average Cosine Similarity {}'.format(cosine_sum))


            # PCA transformation and TSNE
            take_PCA = False                        ### CUIDADO! = False -> no se saca PCA ni se plotea
            if take_PCA:
                n_components=10 #20
                pca_hseq_no_missing = PCA(n_components)
                tsne_hseq_no_missing = TSNE(n_components=2)
                h_seq_pca_tsne_no_missing = tsne_hseq_no_missing.fit_transform(pca_hseq_no_missing.fit_transform(h_seq_no_missing_list))
                h_seq_pca_tsne = tsne_hseq_no_missing.fit_transform(pca_hseq_no_missing.transform(h_seq_list))

                pca_hseq_no_missing = PCA(n_components=2)
                h_seq_pca_no_missing = pca_hseq_no_missing.fit_transform(h_seq_no_missing_list)
                h_seq_pca = pca_hseq_no_missing.transform(h_seq_list)

            missing_index_dict['total_rmse'] = total_rmse
            missing_index_dict['lost_index_rmse'] = lost_index_rmse
            missing_index_dict['total_rmse_no_missing'] = total_rmse_no_missing
            missing_index_dict['lost_index_rmse_no_missing'] = lost_index_rmse_no_missing
            missing_index_dict['cosine_similarity'] = cosine_sum
            if take_PCA:
                missing_index_dict['h_pca_tsne_no_missing'] = h_seq_pca_tsne_no_missing
                missing_index_dict['h_pca_tsne'] = h_seq_pca_tsne
                missing_index_dict['h_pca_no_missing'] = h_seq_pca_no_missing
                missing_index_dict['h_pca'] = h_seq_pca
            missing_index_dict['X_in_no_missing'] = X_in_no_missing_list[:, missing_index]
            missing_index_dict['Y_pred'] = Y_pred_list[:, missing_index]
            missing_index_dict['Y_pred_no_missing'] = Y_pred_no_missing_list[:, missing_index]
            missing_index_dict['X_clean'] = X_clean_list[:, missing_index]
            results_dict['{}'.format(missing_index + 1)] = missing_index_dict

            if plot and take_PCA:
                plt.figure(figsize=(21, 7))
                plt.title('Reconstruction of sensor {}'.format(missing_index + 1))
                plt.plot(X_in_no_missing_list[:, missing_index], label='Noisy Input')
                plt.plot(Y_pred_list[:, missing_index], label='AE output')
                plt.plot(X_clean_list[:, missing_index], label='Clean Data')
                plt.grid()
                plt.legend()
                plt.tight_layout()

                plt.figure(figsize=(21, 7))
                plt.suptitle('Latent space PCA -> TSNE')
                plt.subplot(1, 2, 1)
                plt.scatter(h_seq_pca_tsne_no_missing[:, 0], h_seq_pca_tsne_no_missing[:, 1], label='h')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.subplot(1, 2, 2)
                plt.scatter(h_seq_pca_tsne[:, 0], h_seq_pca_tsne[:, 1], label='h missing')
                plt.legend()
                plt.grid()
                plt.tight_layout()


                plt.figure(figsize=(21, 7))
                plt.suptitle('Latent space PCA')
                plt.subplot(1, 2, 1)
                plt.scatter(h_seq_pca_no_missing[:, 0], h_seq_pca_no_missing[:, 1], label='h')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.subplot(1, 2, 2)
                plt.scatter(h_seq_pca[:, 0], h_seq_pca[:, 1], label='h missing')
                plt.legend()
                plt.grid()
                plt.tight_layout()

                plt.show()

        torch.save(results_dict, self.folder + 'reconstruction_results.pkl')

    def real_time_eval(self, missing=False, n_inputs=6, clean_data_path='data_clean_lin.pkl', plot=False, name=''):  # Checkpoints deben ir ordenados de mayor a menor

        ## data_dict tiene shuffle
        self.new_data_dict = self.processor.process_tanks(noise_power=self.noise_power, noise_inputs=self.noise_inputs, shuffle=self.shuffle_train, 
                                                        folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=self.noise, clean_data=clean_data_path)
        ## data_dict2 no tiene shuffle
        self.new_data_dict2 = self.processor.process_tanks(noise_power=self.noise_power, noise_inputs=self.noise_inputs, shuffle=False,
                                                        folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=self.noise, clean_data=clean_data_path)

        ## Índice
        idx = 2
        upto = 200

        ## Escogemos la porción de datos de testing para hacer la evaluación: test_seq tiene los datos con ruido, clean_seq los originales sin ruido
        test_seq = self.data_dict2['test_data']
        clean_seq = self.data_dict2['test_data_preproc']
        print(test_seq.shape, clean_seq.shape)
        plt.figure(1)
        plt.plot(test_seq[:upto, -1, 2], label='X_clean')
        #plt.plot(clean_seq[:upto, -1, 2], label='X_in_no_missing')
        plt.legend()
        plt.title(f'Indice 2')

        plt.figure(2)
        plt.plot(test_seq[:upto, -1, 0], label='X_clean')
        plt.plot(clean_seq[:upto, -1, 0], label='X_in_no_missing')
        plt.legend()
        plt.title(f'Indice 0')

        plt.figure(3)
        plt.plot(test_seq[:upto, -1, 4], label='X_clean')
        #plt.plot(clean_seq[:upto, -1, 4], label='X_in_no_missing')
        plt.legend()
        plt.title(f'Indice 4')

        plt.show()

        ### Evaluación
        length_eval = self.data_test.shape[0]
        ## Inicia modo de evaluación: model.eval() is a kind of switch for some specific layers/parts of the model that behave differently 
        ## during training and inference (evaluating) time (https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
        self.model.eval() 
        results_dict = {}
        print('\n Evaluating {}'.format(name))

        ## Por cada entrada, recorro una vez el for
        print('\n Analizing index: {}'.format(idx))
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

            # Performances calculation
            total_rmse = root_mean_squared_error(Y_pred_list[:, 2:], X_clean_list[:, 2:])
            lost_index_rmse = root_mean_squared_error(Y_pred_list[:, idx], X_clean_list[:, idx])

            print('Total RMSE: {}'.format(total_rmse))
            print('Lost index RMSE: {}'.format(lost_index_rmse))

            total_rmse_no_missing = root_mean_squared_error(Y_pred_no_missing_list[:, 2:], X_clean_list[:, 2:])
            lost_index_rmse_no_missing = root_mean_squared_error(Y_pred_no_missing_list[:, idx], X_clean_list[:, idx])

            print('Total RMSE no missing: {}'.format(total_rmse_no_missing))
            print('Lost index RMSE no missing: {}'.format(lost_index_rmse_no_missing))


            # Cosine Similarity
            cosine_sum = 0
            for i in range(h_seq_list.shape[0]):
                cosine_sum += cosine_similarity(h_seq_list[i, :].reshape(1, -1), h_seq_no_missing_list[i, :].reshape(1, -1))[0, 0]
            cosine_sum = cosine_sum/h_seq_list.shape[0]

            print('Average Cosine Similarity {}'.format(cosine_sum))

            for i in range(n_inputs):
                missing_index_dict['total_rmse'] = total_rmse
                missing_index_dict['lost_index_rmse'] = lost_index_rmse
                missing_index_dict['total_rmse_no_missing'] = total_rmse_no_missing
                missing_index_dict['lost_index_rmse_no_missing'] = lost_index_rmse_no_missing
                missing_index_dict['cosine_similarity'] = cosine_sum
                missing_index_dict['X_in_no_missing'] = X_in_no_missing_list[:, i]
                missing_index_dict['Y_pred'] = Y_pred_list[:, i]
                missing_index_dict['Y_pred_no_missing'] = Y_pred_no_missing_list[:, i]
                missing_index_dict['X_clean'] = X_clean_list[:, i]
                results_dict['{}'.format(i)] = missing_index_dict
        '''
        torch.save(results_dict, self.folder + 'reconstruction_results.pkl')
        
        ## Importar resultados
        print(f'Load pickle from {self.folder}reconstruction_results.pkl')
        reconstruct_result = torch.load(f'{folder}{name}/reconstruction_results.pkl')
        '''
        reconstruct_result = results_dict

        ## Numpy arrays
        X_clean = reconstruct_result[str(idx)]['X_clean']
        Y_pred = reconstruct_result[str(idx)]['Y_pred']
        total_rmse = reconstruct_result[str(idx)]['total_rmse']
        lost_index_rmse = reconstruct_result[str(idx)]['lost_index_rmse']
        lost_index_rmse_no_missing = reconstruct_result[str(idx)]['lost_index_rmse_no_missing']
        cos_similarity = reconstruct_result[str(idx)]['cosine_similarity']
        X_in_no_missing = reconstruct_result[str(idx)]['X_in_no_missing']
        Y_pred_no_missing = reconstruct_result[str(idx)]['Y_pred_no_missing']

        # Create a 2x2 subplot
        
        idx = 2
        plt.subplot(2, 2, 1)
        plt.title('Reconstruction of input {}'.format(idx))
        plt.plot(Y_pred_no_missing[:upto], label='Y_pred_no_missing')
        plt.plot(Y_pred[:upto], label='Y_pred')
        plt.grid()
        plt.legend()

        idx = 3
        plt.subplot(2, 2, 2)
        plt.title('Reconstruction of input {}'.format(idx))
        plt.plot(Y_pred_no_missing[:upto], label='Y_pred_no_missing')
        plt.plot(Y_pred[:upto], label='Y_pred')
        plt.grid()
        plt.legend()

        idx = 2
        plt.subplot(2, 2, 3)
        plt.title('Reconstruction of input {}'.format(idx))
        plt.plot(Y_pred_list[:upto, idx], label='AE output')
        plt.plot(Y_pred_no_missing_list[:upto, idx], label='Original Data')
        plt.grid()
        plt.legend()

        idx = 3
        plt.subplot(2, 2, 4)
        plt.title('Reconstruction of input {}'.format(idx))
        plt.plot(Y_pred_list[:upto, idx], label='AE output')
        plt.plot(Y_pred_no_missing_list[:upto, idx], label='Original Data')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

                    

if __name__ == '__main__':

    meta = {}
    testing = False
    missing_prob = 0.2
    seqlen = 12 #60
    betas = [1, 1.5]
    folder = 'Results/Nets_60_saltPepper_Addlinear/'
    noise = 'saltPepper'
    noise_powers = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    noise_inputs = False
    if not testing:
        for i in range(len(betas)):
            meta = {'n_inputs': 6, 'hidden_size': 20, 'n_layers': 2, 'bidirectional': False, 'seqlen': seqlen, #hidden_size = 80
                    'tau': 1, 'mode': 'combined', 'noise_power': noise_powers[i], 'noise_inputs':noise_inputs,
                    'batch_size': 64, 'name': 'RNNMAE_80_Addlinear_Mcombined_seqlen_{}_beta_{}_{}'.format(seqlen, betas[i], noise),
                    'missing': True, 'data_type': 'Tanks', 'criterion': 'MAE', 'missing_prob': missing_prob,
                    'beta': betas[i]}

            print('Training {}'.format(meta['name']))
            meta['additional_info'] = 'En este caso no se hace la suma de para bypasear a la cnn'
            save_metadata(meta, '{}{}/'.format(folder, meta['name']))
            meta = load_metadata('{}{}/'.format(folder, meta['name']))
            processor = DataProcessor(seqlen=meta['seqlen'])
            model = RNNAutoencoder_AddLayer(n_inputs=meta['n_inputs'], hidden_size=meta['hidden_size'], n_layers=meta['n_layers'],
                                   bidirectional=meta['bidirectional'], seqlen=meta['seqlen'],
                                   batch_size=meta['batch_size']).to(device)

            training_autoencoder = TrainingAutoencoder(processor, model, data_type=meta['data_type'], shuffle_train=True, noise_inputs=meta['noise_inputs'],
                                                       folder='{}{}/'.format(folder, meta['name']), noise=noise, noise_power=meta['noise_power'],
                                                       path='model', batch_size=meta['batch_size'], missing_prob=meta['missing_prob'])

            training_autoencoder.train(epochs=2, checkpoints=[], beta=meta['beta'], # epochs = 150
                                       missing=meta['missing'], scheduler=False, n_inputs=meta['n_inputs'],
                                       criterion=meta['criterion'], tau=meta['tau'], seqlen=meta['seqlen'], mode=meta['mode'])
            break


    else:
        plot = False #True
        folders = os.listdir(folder)
        folders = [elem for elem in folders if 'fig' not in elem and 'baseline' not in elem and 'svg' not in elem]
        for name in folders:
            meta = load_metadata('{}{}/'.format(folder, name))

            model = RNNAutoencoder_AddLayer(n_inputs=meta['n_inputs'], hidden_size=meta['hidden_size'],
                                            n_layers=meta['n_layers'],
                                            bidirectional=meta['bidirectional'], seqlen=meta['seqlen'],
                                            batch_size=meta['batch_size']).to(device)

            model.load_state_dict(torch.load('{}{}/best_model'.format(folder, meta['name']), map_location=torch.device('cpu')))
            processor = DataProcessor(seqlen=meta['seqlen'])
            training_autoencoder = TrainingAutoencoder(processor, model, data_type=meta['data_type'], shuffle_train=True, noise_inputs=meta['noise_inputs'],
                                                       folder='{}{}/'.format(folder, meta['name']), noise=noise, noise_power=meta['noise_power'],
                                                       path='model', batch_size=meta['batch_size'], missing_prob=meta['missing_prob'])
            #training_autoencoder.eval(missing=meta['missing'], n_inputs=meta['n_inputs'], plot=plot, name=meta['name'])
            