import torch
import pickle
import random
import copy
import pandas as pd
import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def openFile(name):
    with open(name, 'rb') as file:
        return pickle.load(file)

def saveFile(obj, name):
    with open(name, 'wb') as file:
        pickle.dump(obj, file)


def generate_noise(data_clean, multiplier=1, noise_inputs=False, plot=False):
    """
    Generates Salt and Pepper Noise
    """
    if noise_inputs:
        data_clean[:, :2] = data_clean[:, :2]*30
    else:
        inputs = data_clean[:, :2]
        data_clean = data_clean[:, 2:]

    ## CAREFUL! If you change the scaler, you must NOT change it here too
    scaler = StandardScaler()
    scaler.fit(data_clean)
    scales = scaler.scale_

    beta = 0  # the exponent

    # White Noises
    samples = data_clean.shape[0]
    y_white = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(data_clean.shape[1])]
    y_white = np.array(y_white).transpose() * multiplier

    mult = [0.5, 0.7, 1, 1.2, 1.5]
    p = 0.05
    data_salt_and_pepper = copy.deepcopy(data_clean + y_white)

    signs = [-1, 1]
    for j in range(data_clean.shape[0]):
        selected_mults = np.array([random.choice(mult) for j in range(data_clean.shape[1])])
        selected_probs = np.array([int(random.random() < p) for j in range(data_clean.shape[1])])
        selected_signs = np.array([random.choice(signs) for j in range(data_clean.shape[1])])
        addition = scales * selected_mults * selected_probs * selected_signs
        data_salt_and_pepper[j, :] += addition

    # data_salt_and_pepper[np.where(data_salt_and_pepper < 0)] = 0

    if not noise_inputs:
        data_salt_and_pepper = np.concatenate([inputs, data_salt_and_pepper], axis=1)
    else:
        data_salt_and_pepper[:, :2] = data_salt_and_pepper[:, :2]/30

    if plot:
        plt.figure()
        plt.title(multiplier)
        plt.plot(data_salt_and_pepper[1000:1300, 2])
        plt.grid()
        plt.show()
    return data_salt_and_pepper



class DataProcessor():
    """
    Process data from the thickener and from the tanks
    """
    def __init__(self, seqlen=100):
        self.seqlen = seqlen
        self.tickener_signals = ['br_7120_ft_1002', 'bj_7110_ft_1012' , 'bg_7110_dt_1011_solido', 'bk_7110_ft_1030',
            'bp_7110_ot_1003', 'bo_7110_lt_1009_s4', 'bq_7110_pt_1010', 'bi_7110_dt_1030_solido']


    def make_batch(self, data, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        data = np.array(data)

        scaler = RobustScaler(quantile_range=(10, 90))

        data = scaler.fit_transform(data)
        data_out = []
        for i in range(len(data) - seqlen):
            data_out.append(data[i: i + seqlen, :])

        data_out = np.array(data_out)

        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        if shuffle:
            np.random.shuffle(data_out)

        if torch_type:
            data_out = torch.from_numpy(data_out)
            if not cpu:
                data_out = data_out.to(device)


        return data_out.float(), scaler


    def make_batch_tanks(self, data, data_preproc, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        """
                       Window creation
                       In this case each window overlaps in T-1 points with consecutive windows
        """
        scaler = StandardScaler()
        scaler_preproc = StandardScaler()

        data = scaler.fit_transform(data)
        data_preproc = scaler_preproc.fit_transform(data_preproc)
        # print(f'scaler Min and Max = {scaler.data_min_} & {scaler.data_max_}')
        # print(f'scaler_preproc Min and Max = {scaler_preproc.data_min_} & {scaler_preproc.data_max_}')

        data_out = []
        data_out_preproc = []
        for i in range(len(data) - seqlen):
            data_out.append(data[i: i + seqlen, :])
            data_out_preproc.append(data_preproc[i: i + seqlen, :])

        data_out = np.array(data_out)
        data_out_preproc = np.array(data_out_preproc)

        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        if shuffle:
            np.random.shuffle(data_out)
            np.random.shuffle(data_out_preproc)

        if torch_type:
            data_out = torch.from_numpy(data_out)
            data_out_preproc = torch.from_numpy(data_out_preproc)

            if not cpu:
                data_out = data_out.to(device)
                data_out_preproc = data_out_preproc.to(device)

        return data_out.float(), scaler, data_out_preproc.float(), scaler_preproc


    def make_batch_new_data_tanks(self, data, data_preproc, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        """
               Window creation
               In this case each window doesn't overlap with other ones
        """
        data = np.array(data)

        scaler = StandardScaler()
        scaler_preproc = StandardScaler()

        data = scaler.fit_transform(data)
        data_preproc = scaler_preproc.fit_transform(data_preproc)
        # print(f'scaler Min and Max = {scaler.data_min_} & {scaler.data_max_}')
        # print(f'scaler_preproc Min and Max = {scaler_preproc.data_min_} & {scaler_preproc.data_max_}')
        data_out = []
        data_out_preproc = []
        i = 0
        while i < len(data) - seqlen:
            data_out.append(data[i: i + seqlen, :])
            data_out_preproc.append(data_preproc[i: i + seqlen, :])
            i += seqlen

        data_out = np.array(data_out)
        data_out_preproc = np.array(data_out_preproc)

        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        if shuffle:
            np.random.shuffle(data_out)
            np.random.shuffle(data_out_preproc)

        if torch_type:
            data_out = torch.from_numpy(data_out)
            data_out_preproc = torch.from_numpy(data_out_preproc)
            if not cpu:
                data_out = data_out.to(device)
                data_out_preproc = data_out_preproc.to(device)

        return data_out.float(), scaler, data_out_preproc.float(), scaler_preproc


    def make_batch_new_data(self, data, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        """
                     Window creation
                     In this case each window doesn't overlap with other ones
        """
        data = np.array(data)

        scaler = RobustScaler()

        data = scaler.fit_transform(data)
        data_out = []
        i = 0
        while i < len(data) - seqlen:
            data_out.append(data[i: i + seqlen, :])
            i += seqlen

        data_out = np.array(data_out)

        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        if shuffle:
            np.random.shuffle(data_out)

        if torch_type:
            data_out = torch.from_numpy(data_out)
            if not cpu:
                data_out = data_out.to(device)

        return data_out.float(), scaler


    def process_tickener(self, folder='../Thickener_Data/Thickener/', signals=[], ratio=0.8, shuffle=True, new_data=False):
        parameters = openFile(folder + 'parametrosNew.pkl')

        if signals == []:
            signals = self.tickener_signals

        n_inputs = len(signals)

        # Loading Data
        data = pd.read_pickle(folder + 'dcs_data_04_18_2018_TO_26_02_2019_5min.pkl')[signals]


        # Parameters
        centers = np.array([parameters[tag]['center'] for tag in data.columns]).reshape(1, -1)
        scales = np.array([parameters[tag]['scale'] for tag in data.columns]).reshape(1, -1)

        #Limits
        inf_limit = centers - 1*scales
        inf_limit[np.where(inf_limit < 0)] = 0
        sup_limit = centers + 1*scales

        # Physical Restrictions
        def limits_func(variables):
            variables[np.where(variables > sup_limit)] = np.nan
            variables[np.where(variables < inf_limit)] = np.nan
            return variables

        # Apply Physiscal Restictions
        columns = data.columns
        data = pd.DataFrame(limits_func(np.array(data)), columns=columns)

        # NA values filling
        data = data.fillna(method='bfill').fillna(method='ffill')

        if new_data:
            data, scaler = self.make_batch_new_data(data=data, seqlen=self.seqlen, cpu=True,
                                shuffle=shuffle)
        else:
            data, scaler = self.make_batch(data=data, seqlen=self.seqlen, cpu=True, shuffle=shuffle)

        length = data.shape[0]
        data_train = data[:int(length*ratio), :, :]
        data_test = data[int(length*ratio):, :, :]

        data_dict = {'train_data': data_train, 'test_data': data_test, 'scaler': scaler, 'signals': signals}

        return data_dict


    def process_tanks(self, folder='../Tanks_Data/No_Noised_Inputs/', signals=[], ratio=[0.65, 0.2, 0.15], shuffle=True, new_data=False,
                      type_of_noise='white', noise_power=1, noise_inputs=False, clean_data='data_NL_clean.pkl'):
        print(f'Data used: {folder}{clean_data}')
        print(f'noise power = {noise_power}')
        ratio_train = ratio[0]
        ratio_val = ratio[1]
        ratio_test = ratio[2]

        # Loading Data
        ##########################################################
        ## Cambio -> dato linealizados en vez de datos normales ##
        ##########################################################

        ## data_preproc => data original (todo limpio) de tipo np.array
        data_preproc = torch.load(folder + clean_data) # Original data to compare with output data

        ## data => data ruidosa
        if type_of_noise == 'saltPepper':
            print('{} - {}'.format(type_of_noise, noise_power))
            data = generate_noise(data_preproc, multiplier=noise_power, noise_inputs=noise_inputs)
        else:
            data = torch.load(folder + 'data_{}_noise.pkl'.format(type_of_noise))

        ## Uncomment to view the data with and without noise
        # plt.figure(1)
        # for i in range(6):
        #     plt.subplot(int(f'32{i+1}'))
        #     plt.plot(data[:, i], label=f'data_{i}')
        #     plt.plot(data_preproc[:, i], label=f'data_preproc_{i}')
        # plt.legend()
        # plt.show()

        if new_data:
            data, scaler_data, data_preproc, scaler_data_preproc = \
                self.make_batch_new_data_tanks(data=data, data_preproc=data_preproc, seqlen=self.seqlen, cpu=True, shuffle=shuffle)

        else:
            data, scaler_data, data_preproc, scaler_data_preproc = \
                self.make_batch_tanks(data=data, data_preproc=data_preproc, seqlen=self.seqlen, cpu=True, shuffle=shuffle)


        ## Check: no hay NaN values
        has_nan = torch.any(torch.isnan(data))
        if has_nan:
            print(f"The tensor data contains NaN values.")

        data_train = data[:int(data.shape[0] * ratio_train), :, :]
        data_train_preproc = data_preproc[:int(data.shape[0] * ratio_train), :, :]

        data_val = data[int(data.shape[0] * ratio_train):int(data.shape[0] * (ratio_train + ratio_val)), :, :]
        data_val_preproc = data_preproc[int(data.shape[0] * ratio_train):int(data.shape[0] * (ratio_train + ratio_val)), :, :]

        data_test = data[-int(data.shape[0] * ratio_test):, :, :]
        data_test_preproc = data_preproc[-int(data.shape[0] * ratio_test):, :, :]


        data_dict = {'train_data': data_train, 'val_data':data_val, 'test_data': data_test,
                     'train_data_preproc': data_train_preproc, 'val_data_preproc': data_val_preproc,
                     'test_data_preproc': data_test_preproc, 'scaler': scaler_data,
                     'scaler_preproc': scaler_data_preproc, 'signals': signals}
        

        ## Guardar por si acaso en data_dict_processor_{noise_power}.pkl
        torch.save(data_dict, f'{folder}data_dict_processor_{noise_power}.pkl')

        return data_dict

    def fullprocess_tanks(self, folder='../Tanks_Data/No_Noised_Inputs/', shuffle=True, noise_inputs=False,
                      type_of_noise='white', noise_power=1, clean_data='data_NL_clean.pkl'):

        print('Fullprocess_tanks() was used')
        print(f'Data used: {folder}{clean_data}')
        # Loading Data
        ##########################################################
        ## Cambio -> dato linealizados en vez de datos normales ##
        ##########################################################

        ## data_preproc => data original (todo limpio)
        data_preproc = torch.load(folder + clean_data) # Original data to compare with output data
        ## data => data ruidosa
        if type_of_noise == 'saltPepper':
            print('{} - {}'.format(type_of_noise, noise_power))
            data = generate_noise(data_preproc, multiplier=noise_power, noise_inputs=noise_inputs)
        else:
            data = torch.load(folder + 'data_{}_noise.pkl'.format(type_of_noise))

        data, scaler_data, data_preproc, scaler_data_preproc = \
            self.make_batch_tanks(data=data, data_preproc=data_preproc, seqlen=self.seqlen, cpu=True, shuffle=shuffle)

        ## Check: no hay NaN values
        has_nan = torch.any(torch.isnan(data))
        if has_nan:
            print(f"The tensor data contains NaN values.")

        data_dict = {'data': data, 'data_preproc': data_preproc, 'scaler': scaler_data,
                     'scaler_preproc': scaler_data_preproc}

        ## Guardar por si acaso en data_dict_processor_{noise_power}.pkl
        # torch.save(data_dict, f'{folder}data_dict_processor_{noise_power}.pkl')

        return data_dict



if __name__ == '__main__':
    noise_powers = [1, 1.5]
    data_clean = np.zeros(shape=(10000, 6))
    generate_noise(data_clean, multiplier=1.5, noise_inputs=False, plot=False)
    noise_powers = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    noise_inputs = False
    noise = 'saltPepper'
    for noise_power in noise_powers:
        processor = DataProcessor(seqlen=60)
        data_tanks = processor.process_tanks(type_of_noise=noise, shuffle=False, noise_inputs=noise_inputs,
                                             noise_power=noise_power, folder='../Tanks_Data/No_Noised_Inputs/')
        scaler = data_tanks['scaler']
        scaler_preproc = data_tanks['scaler_preproc']
        test_data = scaler.inverse_transform(data_tanks['test_data'][:, -1, :].numpy())[:, 2:]
        test_data_preproc = scaler_preproc.inverse_transform(data_tanks['test_data_preproc'][:, -1, :].numpy())[:, 2:]
        rmse = np.sqrt(mean_squared_error(test_data, test_data_preproc))

        print('Noise Power: {} | RMSE: {}'.format(noise_power, rmse))