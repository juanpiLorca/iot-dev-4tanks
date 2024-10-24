import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_seq_contrastive_loss(h, augmentation_factor=1, tau=1, mode='combined'):
    '''
    NCE LOSS calculation
    mode = 'seq' -> Only Consecutive elements (Zc)
    mode = 'random' -> Only Random Selected elements (Zr)
    mode = 'combined' -> Combined elements (Zc + Zr)
    '''

    ## Elimino de h las componentes que tengas una sola dimensión
    ## Dimensión de h: batch_size, n_layers (1, n_layers*batch_size, n_layers al comienzo)
    h = h.squeeze()
    batch_size = h.shape[0]

    ### MASK
    # augmentation_factor = 1 por default (se mantiene en la implementación)
    # mask = matriz identidad de tamaño batch_size, batch_size
    mask = np.eye(augmentation_factor*batch_size)

    # Modo ultilizado: 'combined'
    if mode == 'seq' or mode == 'combined':
        # l_past: matriz identidad con la línea uno (k) arriba de la diagonal
        # l_future: matriz identidad con la línea uno (k) abajo de la diagonal
        l_past = np.eye(augmentation_factor*batch_size, k=1)
        l_future = np.eye(augmentation_factor*batch_size, k=-1)
        # mask_past: matriz llena de unos excepto la diagonal y su traza superior (con ceros)
        # mask_future: matriz llena de unos excepto la diagonal y su traza inferior (con ceros)
        mask_past = 1 - (mask + l_past)
        mask_future = 1 - (mask + l_future)
        mask_past = torch.from_numpy(mask_past).to(device).bool()
        mask_future = torch.from_numpy(mask_future).to(device).bool()

    # mask: matriz llena de unos excepto la diagonal (ceros)    
    mask = 1 - mask
    mask = torch.from_numpy(mask).bool().to(device)

    ### Cosine similarity: de la matriz de hidden_states h
    h_norm = h.norm(dim=1)[:, None]
    # h1_norm y h2_norm: tienen dimensiones de h, es decir, batch_size, n_layers
    h1_norm = h / h_norm
    h2_norm = h / h_norm
    # similarities: tiene dimensión batch_size, batch_size
    similarities = torch.mm(h1_norm, h2_norm.transpose(0, 1))

    ### Loss: recorro la matriz de similaridades y aplico                                      exp(sim[i, j] / tau)
    #                                                         loss(i, j, q) =   -log( ---------------------------------------- )
    #                                                                                 sum( mask[i, :] * exp(sim[i, :] / tau) )
    #  
    #   Funcionamiento:
    #                   i) mask = mask_past: la máscara elimina el dato de la columna i e i+1 => mido similaridad sin considerar el
    #                                 dato actual y el siguiente  
    #                   ii) mask = mask_future: la máscara elimina el dato de la columna i e i-1 => mido similaridad sin considerar el  
    #                                    dato actual y el anterior
    #
    #   Consecuencia: si quiero que la pérdida se reduzca lo más posible, necesito que el argumento de-log() sea muy grande, es decir, la  
    #                 sumatoria del denominador debe ser pequeña y la similaridad del numerador grande                                                 
    def loss(i,j, mask):
        l_ij = - torch.log(torch.exp(similarities[i, j]/tau)/(torch.sum(mask[i, :]*torch.exp(similarities[i, :]/tau))))
        return l_ij

    final_loss = 0
    if mode == 'random':
        n = batch_size - 1
        p = 2
        for i in range(0, n, p):
            final_loss += loss(i, i + 1, mask) + loss(i + 1, i, mask)

    elif mode == 'seq':
        n = batch_size - 1
        p = 1
        for i in range(0, n, p):
            final_loss += loss(i, i + 1, mask_future) + loss(i + 1, i, mask_past)

    # mode == 'combined'
    else: 
        n = int(batch_size/2) - 1
        p = 1

        for i in range(0, n, p):
            # Tomamos los elementos sobre y bajo la diagonal (los [i,j]) y aplicamos la máscara:
            #       i) loss(i, i + 1, mask_future) -> toma el elemento a la derecha de la diagonal = (1,2), (2,3), etc y 
            #          toma su similaridad. Este valor queremos que sea grande. Por otro lado, en el denominador nos quedamos
            #          la fila i de la matriz de similaridad, pero eliminando los términos i e i-1 (solo queda el término "futuro").
            #          Queremos que la suma de estas similaridades sea pequeña, es decir, que el término futuro no se asimile al
            #          resto de los términos de la secuencia
            #       ii) loss(i + 1, i, mask_past) -> toma el elemento a la izquierda de la diagonal = (2,1), (3,2), etc y 
            #          toma su similaridad. Este valor queremos que sea grande. Por otro lado, en el denominador nos quedamos
            #          la fila i de la matriz de similaridad, pero eliminando los términos i e i+1 (solo queda el término "pasado").
            #          Queremos que la suma de estas similaridades sea pequeña, es decir, que el término pasado no se asimile al
            #          resto de los términos de la secuencia
            final_loss += loss(i, i + 1, mask_future) + loss(i + 1, i, mask_past)

    final_loss = final_loss/(2*(n + 1)/p)

    return final_loss

