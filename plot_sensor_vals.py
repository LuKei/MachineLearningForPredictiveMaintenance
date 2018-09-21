import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


with open('MachineData/' + 'train_FD001.csv', newline='') as csvfile:
    df_train = pd.read_csv(csvfile, header=None, names=range(26))

df_train_unit0 = df_train[df_train[0] == 1]
df_train_unit0.drop(labels=[0,1,2,3,4], axis='columns', inplace=True)

def plot_sensors(df_train_unit_x):
    #Alle Sensoren zur ersten unit number plotten
    plt.figure(figsize=(20, 20))
    i = 1
    xlabel_indices = [19,20,21]
    ylabel_indices = [1,4,7,10,13,16,19]
    for column in df_train_unit_x.columns:
        plt.subplot(7,3,i)
        plt.gca().tick_params(labelsize=14)
        plt.plot(df_train_unit0[column])
        if i in xlabel_indices:
            plt.xlabel('Zeit in Zyklen', fontsize=16)
        if i in ylabel_indices:
            plt.ylabel('Sensor-Wert', fontsize=16)
        plt.title('Sensor ' + str(i), fontsize=16)
        i += 1

    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.19, right=0.82, hspace=0.52,
                        wspace=0.2)

    #plt.show()

    plt.savefig(fname='Abbildungen/Sensoren.png', format='png')


def plot_sensor8_with_exp(df_train_unit_x):
    plt.figure(figsize=(20, 20))

    # Geschätzte Exponentialfunktion plotten
    x = np.linspace(0, 200, 10000)
    # plt.ylim(0, 0.35)
    # plt.xlim(0, 200)
    plt.plot(x, (0.67*np.exp(0.025*x-1.2) /100) + 2388.05)


    # Sensor 8 der ersten unit number plotten
    plt.plot(df_train_unit_x[12])


    plt.show()

plot_sensor8_with_exp(df_train_unit0)