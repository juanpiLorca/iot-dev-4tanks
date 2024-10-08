import matplotlib.pyplot as plt 
import pandas as pd

files = ["plant_client.csv", "autoencoder.csv", "controller.csv"]
paths = [f"results/{file}" for file in files]
variables = [
    ["t", "dt", "x1", "x2", "x3", "x4"], 
    ["t", "dt", "xf1", "xf2", "xf3", "xf4"], 
    ["t", "dt", "u1", "u2", "u3", "u4"]
]

def plotter(x, names):
    t = x[0]
    dt = x[1]
    nrows = len(x)
    colors = ["b", "r", "g", "c"]
    fig, ax = plt.subplots(nrows=(nrows-2))
    num = 0
    for j in range(2, nrows):  
        ax[j-2].clear()
        ax[j-2].plot(t, x[j], label=names[j], color=colors[num])
        ax[j-2].set_xlabel('Time [s]')
        ax[j-2].set_ylabel('Magnitude')
        ax[j-2].grid()
        ax[j-2].legend(loc='upper right', bbox_to_anchor=(1.2, 1)) 
        num += 1
    fig.tight_layout()  
    plt.subplots_adjust(hspace=0.7)  
    return fig, ax

def plot_results(data, variables): 
    for x, names in zip(data, variables): 
        _, _ = plotter(x, names)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    data_client_plant = pd.read_csv(paths[0])
    data_autoencoder = pd.read_csv(paths[1])
    data_controller = pd.read_csv(paths[2])
    data = [
        [data_client_plant[name_var] for name_var in variables[0]],
        [data_autoencoder[name_var] for name_var in variables[1]],
        [data_controller[name_var] for name_var in variables[2]]
    ]
    plot_results(data, variables)

 