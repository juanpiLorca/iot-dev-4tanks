import torch
import copy
import random
import time
import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt
from utils.nets import RNNAutoencoder_AddLayer
from utils.data_processor import DataProcessor, generate_noise
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.random_addlinear_noise import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder():
    def __init__(self, folder='NL_AE_models/', noise='saltPepper', name='uut_noise2'):
        ## folder = carpeta donde se guardan todas las pruebas
        self.folder = folder
        ## noise = tipo de ruido, con posibilidad: 'saltPepper' o else (no implementado)
        self.noise = noise
        ## Folder name del modelo
        self.name = name

        meta = load_metadata('{}{}/'.format(folder, name))
        self.meta = meta
        print(f'Testing {meta["name"]} with noise power = {meta["noise_power"]}')
            
        self.model = RNNAutoencoder_AddLayer(n_inputs=meta['n_inputs'], hidden_size=meta['hidden_size'],
                n_layers=meta['n_layers'],
                bidirectional=meta['bidirectional'], seqlen=meta['seqlen'],
                batch_size=meta['batch_size']).to(device)
        
        self.model.load_state_dict(torch.load('{}{}/model'.format(folder, meta['name']), map_location=torch.device('cpu')))
        
        self.processor = DataProcessor(seqlen=meta['seqlen'])
        
        # testing_autoencoder = TestingAutoencoder(processor, model, data_type=meta['data_type'], shuffle_train=True, noise_inputs=meta['noise_inputs'],
        #             folder='{}{}/'.format(folder, meta['name']), noise=noise, noise_power=meta['noise_power'],
        #             path='model', batch_size=meta['batch_size'], missing_prob=meta['missing_prob'])
        

        # testing_autoencoder.eval(missing=meta['missing'], n_inputs=meta['n_inputs'], plot=plot, name=meta['name'])

        self.buff = torch.randn(1, meta['seqlen'], meta['n_inputs'])

        self.data = self.processor.process_tanks(noise_power=meta['noise_power'], noise_inputs=meta['noise_inputs'], shuffle=False,
                                                            folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=noise, clean_data='data_NL_clean.pkl')

        self.Y_pred_list = []

    def test_sequence(self):

        meta = self.meta
        noise = self.noise
        name = self.name
        model = self.model
        batch_size = 1 #meta['batch_size']

        ## data_dict tiene shuffle
        data_dict = self.processor.process_tanks(noise_power=meta['noise_power'], noise_inputs=meta['noise_inputs'], shuffle=True, 
                                                            folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=noise, clean_data='data_NL_clean.pkl')
        ## data_dict2 no tiene shuffle
        data_dict2 = self.processor.process_tanks(noise_power=meta['noise_power'], noise_inputs=meta['noise_inputs'], shuffle=False,
                                                            folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=noise, clean_data='data_NL_clean.pkl')

        data_train = data_dict['train_data']
        data_test = data_dict2['test_data']
        scaler = data_dict['scaler']
        scaler_preproc = data_dict['scaler_preproc']

        ## Escogemos la porción de datos de testing para hacer la evaluación: test_seq tiene los datos con ruido, clean_seq los originales sin ruido
        test_seq = data_dict2['test_data']
        clean_seq = data_dict2['test_data_preproc']

        ### Evaluación
        length_eval = data_test.shape[0]
        ## Inicia modo de evaluación: model.eval() is a kind of switch for some specific layers/parts of the model that behave differently 
        ## during training and inference (evaluating) time (https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
        model.eval() 
        print('\n Evaluating {}'.format(name))
        ## Por cada entrada, recorro una vez el for
        
        missing_index = 2
        print('\n Analizing missing index: {}'.format(missing_index + 1))
        p = 0

        X_in_list = []
        Y_pred_list = []
        X_clean_list = []

        with torch.no_grad():
            ## length_eval es el número total de columnas de data en testeo
            # print(f'test_seq = {test_seq.shape}')
            ## If you want to test a subset of the testing, make the length_eval variable shorter
            length_eval = 10**4
            cnt = 0

            while p < length_eval:
                ## X_in es el batch analizado, que va desde la columna p hasta p+batch_size y contiene datos con ruido
                X_in = test_seq[p:p + batch_size, :, :].to(device)

                ## Print to check
                cnt += 1
                # if cnt <= 300:
                    # print(f'________________________________ Iteracion {cnt} ________________________________')
                    # print(X_in)
                if cnt%1000 == 0:
                    print(f'count = {cnt}')

                # print(f'X_in = {X_in.shape}')

                ## X_clean es el batch analizado, que va desde la columna p hasta p+batch_size
                X_clean = clean_seq[p:p + batch_size, :, :]
                ## Actualizamos el batch count (puntero)
                p += batch_size

                ### Predicciones: detalle en línea 239
                Ypred, _, _ = model(X_in)
                # print(f'Ypred = {Ypred.shape}')

                # Listas
                X_in_list.append(X_in[:, -1, :].cpu().numpy())
                Y_pred_list.append(Ypred[:, -1, :].cpu().numpy())
                X_clean_list.append(X_clean[:, -1, :].cpu().numpy())

        # Concatenación y escalamiento de los resultados
        X_in_list = scaler.inverse_transform(np.concatenate(X_in_list, axis=0))
        Y_pred_list = scaler.inverse_transform(np.concatenate(Y_pred_list, axis=0))
        X_clean_list = scaler_preproc.inverse_transform(np.concatenate(X_clean_list, axis=0))
        
        ## Noisy signal
        Y_noisy = generate_noise(X_clean_list, multiplier=1, noise_inputs=False, plot=False)
        print(Y_noisy.shape)

        upto = 10000
        for i in range(1, meta['n_inputs'] + 1):
            plt.figure(1)
            plt.subplot(int(f'32{i}'))
            plt.plot(Y_noisy[:upto, i-1], label='Y_noisy')
            plt.plot(Y_pred_list[:upto, i-1], label='Y_pred')
            plt.plot(X_clean_list[:upto, i-1], label='Y_clean')
            # plt.plot(X_in_no_missing[:upto], label='X_in_no_missing')
            plt.legend(loc='upper right')
            plt.title(f'Output {i}')

            # plt.figure(2)
            # plt.subplot(int(f'32{i}'))
            # plt.plot(X_clean_list[:upto, i-1], label='X_clean')
            # plt.legend()
            # plt.title(f'Indice {i}')

        plt.show()

    def buffer(self, data_point):
        ## model_input es la lista de tamaño seqlen que ingresa al modelo y es predicha
        meta = self.meta
        model = self.model

        ### Evaluación
        ## Inicia modo de evaluación: model.eval() is a kind of switch for some specific layers/parts of the model that behave differently 
        ## during training and inference (evaluating) time (https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
        model.eval() 

        with torch.no_grad():
            # print(f'buff = {self.buff.shape}')

            ## Buffer for data list. The result must be a tensor of length seqlen and 6 inputs
            ## data_point is a numpy array of size 1 x n_inputs

            # data point to tensor of shape 1 x seqlen x n_inputs
            data_point = torch.from_numpy(data_point).unsqueeze(1) 
            # print(f'data_point = {data_point}')

            # Shift tensor buffer one space to the left
            model_input = torch.cat((self.buff[:, 1:, :], data_point), dim=1)
            model_input = model_input.to(torch.float32)
            self.buff = model_input
            # print(f'buff = {self.buff[:, 57:, :]}')

            ### Predicciones
            Ypred, _, _ = model(model_input)
            # print(f'Ypred = {Ypred}')
            # print(f'Ypred = {Ypred.shape}')

            # New filtered point form AE
            new_point = Ypred[:, -1, :].cpu().numpy()
            # print(f'new_point = {new_point.shape}') # new_point.shape = (1, 6)
            # if self.Y_pred_list.shape[0] == 0:
            #     self.Y_pred_list = new_point
            # else:
            self.Y_pred_list.append(new_point)

    def scale(self):
        scaler = self.data['scaler']
        print(f'scaler = {scaler}')
        self.Y_pred_list = scaler.inverse_transform(np.concatenate(self.Y_pred_list, axis=0))

    def noise_datapoint(self, data_clean, multiplier_white=1, multiplier_SP=1):
            """
            Generates Salt and Pepper Noise
            """

            scaler = StandardScaler()
            scaler.fit(data_clean)
            scales = scaler.scale_

            ## White Noises
            beta = 0  # the exponent
            samples = data_clean.shape[0]
            # To avoid error, the code generates 2 random white noises and selects only the first one
            y_white = [cn.powerlaw_psd_gaussian(beta, 2) for i in range(data_clean.shape[1])]
            y_white = np.array(y_white).transpose() * multiplier_white
            y_white = y_white[0]

            mult = [0.5, 0.7, 1, 1.2, 1.5]
            p = 0.05
            data_salt_and_pepper = copy.deepcopy(data_clean + y_white)

            signs = [-1, 1]
            for j in range(data_clean.shape[0]):
                selected_mults = np.array([random.choice(mult) for j in range(data_clean.shape[1])])
                selected_probs = np.array([int(random.random() < p) for j in range(data_clean.shape[1])])
                selected_signs = np.array([random.choice(signs) for j in range(data_clean.shape[1])])
                addition = scales * selected_mults * selected_probs * selected_signs
                data_salt_and_pepper[j, :] += addition * multiplier_SP

            return data_salt_and_pepper


if __name__ == '__main__':
    start_time = time.time()
    noise='saltPepper'
    name='uut_noise2'
    meta = load_metadata('{}{}/'.format(folder, name))
    processor = DataProcessor(seqlen=60)

    data_dict2 = processor.process_tanks(noise_power=meta['noise_power'], noise_inputs=meta['noise_inputs'], shuffle=False,
                                                                folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=noise, clean_data='data_NL_clean.pkl')
    test_data = data_dict2['test_data']
    clean_data = data_dict2['test_data_preproc']
    print(test_data.shape)

    scaler_preproc = data_dict2['scaler_preproc']
    np_clean_data = clean_data.numpy()
    np_clean_data = scaler_preproc.inverse_transform(np_clean_data[:, 0, :])
    clean_data = torch.from_numpy(np_clean_data)
    print(clean_data.shape)

    AE = AutoEncoder()
    # data = AE.data['test_data'] # torch.Size([7191, 60, 6])

    ## Test
    # AE.test_sequence()
    # print('Test finished')
    
    upto = 10000
    for i in range(upto):
        ## Select the data that is taken to make the sample
        # print(f'________________________________ Iteracion {i+1} ________________________________')
        # print(f'test_data = {test_data.shape}')
        np_test_data = test_data[i, :, :].numpy()
        # print(np_test_data.shape)
        point = np_test_data[0, :].reshape(1,6)
        # print(f'point = {point}')
        AE.buffer(data_point=point)

    AE.scale()

    # print('\nEnd of for loop')
    # print(f'Y_pred_list = {AE.Y_pred_list} \nshape = {AE.Y_pred_list.shape}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {int(elapsed_time)} seconds")

    Y_pred_array = np.array([])
    for i in range(len(AE.Y_pred_list)):
        if Y_pred_array.shape[0] == 0:
            Y_pred_array = AE.Y_pred_list[i]
        else:
            Y_pred_array = np.vstack((Y_pred_array, AE.Y_pred_list[i]))
    print(f'Y_pred_array = {Y_pred_array.shape}')
    torch.save(Y_pred_array, 'Y_pred_array.pkl')
    Y_pred_array = torch.load('Y_pred_array.pkl')

    ## Convert test_data (noisy) back to the original range
    scaler = data_dict2['scaler']
    np_test_data = test_data.numpy()
    np_test_data = scaler.inverse_transform(np_test_data[:, 0, :])
    test_data = torch.from_numpy(np_test_data)
    print(test_data.shape)

    for i in range(1,5):        
        plt.figure(2)
        plt.subplot(int(f'22{i}'))
        plt.plot(test_data[:upto, i+1], label='Y_noisy')
        plt.plot(Y_pred_array[:, i+1], label='Y_AE')
        plt.plot(clean_data[:upto, i+1], label='Y_clean')
        plt.legend()
        plt.title(f'x_{i-1}')
        plt.legend(loc='upper left')
    plt.show()

    ## RMSE
    buff = int(meta['seqlen'] * 1.5)
    print(Y_pred_array[buff:upto, 2:].shape, clean_data[buff:upto, 2:].shape)
    rmse_pred_clean = np.sqrt(mean_squared_error(Y_pred_array[buff:upto, 2:], clean_data[buff:upto, 2:]))
    print(f'RMSE to clean states = {rmse_pred_clean:.3f}')
