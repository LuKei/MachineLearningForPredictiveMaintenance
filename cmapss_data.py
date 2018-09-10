import math
import numpy as np
import pandas as pd

def get_sensor_data(filename_train, filename_test, filename_RUL, maximum_RUL=90):
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    columns_to_exclude = [0,1,2,3,4,5,6,7,9,10,13,14,18,19,20,21,22,23,24,25]
    #columns_to_exclude = [0,1,2,3,4,5,9,10,11,12,13,14,16,17,18,20,21,22,23,24,25]
    with open('MachineData/' + filename_train, newline='') as csvfile:
        df = pd.read_csv(csvfile, header = None, names=range(26))

        y_train = []
        split_indices = []
        last_unit_num = df[0].iloc[0]
        for index, row in df.iterrows():
            if row[0] != last_unit_num:
                split_indices.append(index)
                last_unit_num = row[0]
        df_list = np.split(df, split_indices)
        for df_item in df_list:
            df_row_size = df_item.shape[0]
            index = 0
            for row in df_item.itertuples():
                y_train.append(min(df_row_size - index - 1, maximum_RUL))
                index += 1
        y_train = np.array(y_train)

        df.drop(labels=columns_to_exclude, axis='columns', inplace=True)

        #Spalten, die nur einen Wert haben entfernen
        # for col in df.columns:
        #     if len(df[col].unique().tolist()) < 2:
        #         columns_to_exclude.append(col)
        # df.drop(labels=columns_to_exclude, axis='columns', inplace=True)

        X_train = df.values

    with open('MachineData/' + filename_test, newline='') as csvfile:
        df = pd.read_csv(csvfile, header=None, names=range(26))

        indices_of_last_values = []
        for unit_num in df[0].unique():
            indices_of_last_values.append(df[df[0] == unit_num].index[-1])

        df.drop(labels=columns_to_exclude, axis='columns', inplace=True)

        X_test = df.ix[indices_of_last_values].values

    with open('MachineData/' + filename_RUL, newline='') as csvfile:
        df = pd.read_csv(csvfile, header=None)

        y_test = []
        for row in df.itertuples():
            y_test.append(row[1])

        y_test = np.array(y_test)


    return X_train, X_test, y_train, y_test



