import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_Lout_conv(Lin=20,kernel=1, padding=0, dilation=1, stride=1):
    return (Lin + 2*padding - dilation*(kernel - 1)- 1)/stride + 1


class RNNAutoencoder(nn.Module):
    def __init__(self, n_inputs=8, hidden_size=20, bidirectional=False, n_layers=1, seqlen=60, batch_size=64):
        super(RNNAutoencoder, self).__init__()
        self.n_inputs = n_inputs
        n_outputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.teacher_forcing_prob = 1
        self.period_decay = 50
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.enc_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size,
                              batch_first=True, bidirectional=bidirectional)
        self.dec_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size*self.directions, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size*self.directions, out_features=n_outputs)

        for name, param in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(param.data)

    def decay_teacher(self, it):
        self.teacher_forcing_prob = max(0., 1 - it*(0.8/self.period_decay))


    def forward(self, x):
        # Encoder
        size = x.shape
        #print(f'Encoder: x.shape = {size}')
        h = self.init_hidden(size[0])
        _, h = self.enc_gru(x, h)
        #print(f'Encoder: h.shape = {h.shape}')

        # Decoder
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers,  size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size*self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1, self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h


    def encoder(self, x):# Encoder
        size = x.shape
        h = self.init_hidden(size[0])
        _, h = self.enc_gru(x, h)
        return x, h

    def decoder(self, x, h):
        # Decoder
        size = x.shape
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers, size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size * self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1,
                                      self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h

    ## Inicializar pesos random
    def init_hidden(self, batch_size):
        return torch.randn(self.n_layers * self.directions, batch_size, self.hidden_size).float().to(
            self.enc_gru.all_weights[0][0].device)


## Modelo utilizado en testeo
class RNNAutoencoder_AddLayer(nn.Module):
    def __init__(self, n_inputs=8, hidden_size=20, bidirectional=False, n_layers=1, seqlen=60, batch_size=64):
        super(RNNAutoencoder_AddLayer, self).__init__()
        self.n_inputs = n_inputs
        n_outputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.teacher_forcing_prob = 1
        self.period_decay = 50
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        
        ### Layers: definimos las capas de la red
        # Encoder
        self.enc_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size,
                              batch_first=True, bidirectional=bidirectional)
        # Decoder
        self.dec_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size*self.directions, batch_first=True)
        # Capa lineal posterior al decoder (salida entra a Lae)
        self.linear = nn.Linear(in_features=hidden_size*self.directions, out_features=n_outputs)
        # Capa lineal entre hiddden state h(T) (entre encoder y decoder) y segunda capa lineal
        self.linear_nce1 = nn.Linear(hidden_size, int(hidden_size/4))
        # Segunda capa lineal posterior a primera capa lineal (salida a L_nce)
        self.linear_nce2 = nn.Linear(int(hidden_size/4), int(hidden_size/4))

        ## Los parámetros se inicializan con parametros normalizados
        for name, param in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(param.data)

    ## Función estrictamente decreciente que satura en cero
    def decay_teacher(self, it):
        self.teacher_forcing_prob = max(0., 1 - it*(0.8/self.period_decay))


    def forward(self, x):
        ### Encoder
        # size = (batch_size, seqlen, n_inputs)
        size = x.shape
        # Inicializar hidden state random
        h = self.init_hidden(size[0])
        # Pasar data x por encoder y retornar hiddenstate h (h.shape = n_layers, batch_size, hidden_size)
        _, h = self.enc_gru(x.to(device), h.to(device))

        # Pasos aquí:
        #       1. hidden state h pasa por la capa lineal nce1
        #       2. se aplica la función no lineal relu() a la salida anterior
        #       3. la salida anterior entra a la segunda capa lineal nce2
        #       4. el output final es h_nce
        h_nce = self.linear_nce2(torch.relu(self.linear_nce1(h[-1, :, :])))

        ### Decoder
        outs = []
        # teacher_forcing_prob es una función decreciente que depende del número de iteraciones. La idea es que, al comienzo,
        # 'use_teacher_forcing' sea True (porque es un número grande). Luego, a medida que se entrena la red, 'use_teacher_forcing'
        # se achica y en algún punto será siempre False. Qué sucede en cada caso, se describe a continuación
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        ##### DUDA: No entiendo bien qué pasa abajo. En la práctica, el tamaño disminuye de [128, 20, 6] a [64, 20, 6] (batch_size, seqlen, n_inputs)

        ## Caso True: En este caso, tomo todo x (la ventana de seqlen completa) y lo paso por la red
        if use_teacher_forcing:
            # Recibo h y hago reshape a tamaño (2*batchsize, seqlen, n_inputs)
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers,  size[0], -1))

            # Se pasa la salida del decoder por la capa lineal
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size*self.directions))
            outs.append(x_out)

        ## Caso False: en este caso, tomo el primer elemento de x, lo paso por la red y lo que sale, lo paso de nuevo. Esto se repite seqlen veces
        else:
            # Se hace reshape para agregar la dimension 1, si no queda de shape (size[0], self.n_inputs)
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs) 
            # Se hace reshape de h para que quede de: n_layers, batch_size, -1 (hidden_size)
            h_out = h.reshape(self.n_layers, size[0], -1)
            # Ciclo que se repite seqlen veces
            for i in range(size[1]):
                # Paso x (de un t particular) por el decoder
                x_out, h_out = self.dec_gru(x_out.to(device), h_out.to(device))
                # Le hago reshape para volver al tamaño correcto: batch_size, 1, hidden_size
                x_out = x_out.reshape(size[0], 1, self.hidden_size * self.directions)   # Saul:  La primera dimensión debería ser size[0]*size[1] 
                                                                                        #        pero en este caso size[1] es 1
                
                # Se pasa la salida del decoder por la capa lineal
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)
        #print(f'outs = {outs.shape}')

        return outs, h_nce, h


    ### Esto NO es necesario, porque el Encoder y Decoder ya están explícitos en el forward()

    def encoder(self, x):# Encoder
        size = x.shape
        h = self.init_hidden(size[0])
        print(f' Encoder func: h = {h.shape}')
        _, h = self.enc_gru(x, h)
        print(f' Encoder func2: h = {h.shape}')
        h_nce = self.linear_nce2(torch.tanh(self.linear_nce1(h[-1, :, :])))
        print(f' Encoder func: h_nce = {h_nce.shape}')
        return x, h_nce

    def decoder(self, x, h):
        # Decoder
        size = x.shape
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers, size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size * self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1,
                                      self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h

    ### Inicializar pesos random al comenzar
    def init_hidden(self, batch_size):
        return torch.randn(self.n_layers * self.directions, batch_size, self.hidden_size).float().to(
            self.enc_gru.all_weights[0][0].device)



