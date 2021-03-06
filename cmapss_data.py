import math
import numpy as np
import pandas as pd

columns_to_exclude = [0,1,2,3,4,5,6,7,9,10,13,14,18,19,20,21,22,23,24,25]


def get_train_data(filename_train, maximum_RUL=99999):
    X_train = None
    y_train = None
    with open('MachineData/' + filename_train, newline='') as csvfile:
        df_train = pd.read_csv(csvfile, header = None, names=range(26))

        y_train = []
        split_indices = []
        last_unit_num = df_train[0].iloc[0]
        for index, row in df_train.iterrows():
            if row[0] != last_unit_num:
                split_indices.append(index)
                last_unit_num = row[0]
        df_list = np.split(df_train, split_indices)
        for df_item in df_list:
            df_row_size = df_item.shape[0]
            index = 0
            for row in df_item.itertuples():
                y_train.append(min(df_row_size - index - 1, maximum_RUL))
                index += 1
        y_train = np.array(y_train)

        df_train.drop(labels=columns_to_exclude, axis='columns', inplace=True)

        X_train = df_train.values

    return X_train, y_train


def get_valid_test_data(filename_valid_train, filename_RUL, window_size=4, maximum_RUL = 90, stop_at_RUL = 99999):
    X_valid_test = None
    y_valid_test = []

    with open('MachineData/' + filename_valid_train, newline='') as csvfile:
        with open('MachineData/' + filename_RUL, newline='') as csvfile_RUL:
            df = pd.read_csv(csvfile, header=None, names=range(26))
            df_valid_test = pd.DataFrame(columns=df.columns, dtype=float)

            df_RUL = pd.read_csv(csvfile_RUL, header=None)

            idx_RUL = 0
            for unit_num in df[0].unique():

                df_unit_num = df[df[0] == unit_num]
                RUL = df_RUL.iloc[idx_RUL][0] + ((window_size-1) / 2)

                for j in range(0, int(df_unit_num.shape[0] / window_size)):
                    if j < stop_at_RUL / window_size:
                        desired_slice = slice(df_unit_num.shape[0]-(window_size*(j+1)),df_unit_num.shape[0]-(window_size*j))
                        mean_window_series = df_unit_num.iloc[desired_slice].sum(axis='index') / window_size
                        df_valid_test = df_valid_test.append(mean_window_series, ignore_index=True)

                        y_valid_test.append(min(RUL, maximum_RUL))
                        RUL += window_size

                idx_RUL += 1


            df_valid_test.drop(labels=columns_to_exclude, axis='columns', inplace=True)

            X_valid_test = df_valid_test.values
            y_valid_test= np.array(y_valid_test)

    return X_valid_test, y_valid_test
