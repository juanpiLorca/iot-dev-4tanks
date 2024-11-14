import matplotlib.pyplot as plt 
import pandas as pd


files = ["plant_client.csv", "autoencoder.csv", "controller.csv"]
paths = [f"{file}" for file in files]
variables = [
    [
        "t", "dt", "x1", 
        "x2", "x3", "x4"
    ], 
    [
        "t", "dt",  
        "xn1", "xn2", 
        "xn3", "xn4", 
        "xf1", "xf2", 
        "xf3", "xf4"
    ], 
    [
        "t", "dt", "u1", "u2"
    ]
]


data_client_plant = pd.read_csv(paths[0])
data_autoencoder = pd.read_csv(paths[1])
data_controller = pd.read_csv(paths[2])
data = [
    [data_client_plant[name_var] for name_var in variables[0]],
    [data_autoencoder[name_var] for name_var in variables[1]],
    [data_controller[name_var] for name_var in variables[2]]
]
 
t = data_client_plant["t"]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].plot(t, data_client_plant["x1"], label=r"$x_1$")
axs[0,0].plot(t, data_autoencoder["xf1"], label=r"$\hat{x}_1$")
axs[0,0].plot(t, data_autoencoder["xn1"], label=r"$x_1^{noised}$")
axs[0,0].plot(t, data_controller["ref"], linestyle="--", label=r"$x_{1,ref}$")
axs[0,0].set_xlabel("Time [s]")
axs[0,0].set_ylabel(r"$x_1(t)$")
axs[0,0].legend()
axs[0,0].grid()
axs[0,1].plot(t, data_client_plant["x2"], label=r"$x_2$")
axs[0,1].plot(t, data_autoencoder["xf2"], label=r"$\hat{x}_2$")
axs[0,1].plot(t, data_autoencoder["xn2"], label=r"$x_2^{noised}$")
axs[0,1].plot(t, data_controller["ref"], linestyle="--", label=r"$x_{2,ref}$")
axs[0,1].set_xlabel("Time [s]")
axs[0,1].set_ylabel(r"$x_2(t)$")
axs[0,1].legend()
axs[0,1].grid()
axs[1,0].plot(t, data_client_plant["x3"], label=r"$x_3$")
axs[1,0].set_xlabel("Time [s]")
axs[1,0].set_ylabel(r"$x_3(t)$")
axs[1,0].legend()
axs[1,0].grid()
axs[1,1].plot(t, data_client_plant["x4"], label=r"$x_4$")
axs[1,1].set_xlabel("Time [s]")
axs[1,1].set_ylabel(r"$x_4(t)$")
axs[1,1].legend()
axs[1,1].grid()
plt.tight_layout()
plt.savefig("local_results.jpg", format="jpg")
plt.show()