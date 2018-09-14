import csv
import numpy as np
import rdflib
import pandas as pd


def get_sensor_data_array(filename, machine_id, anomalies='include', anomaly_filename=None):
    csv_list = []

    if anomalies != 'include':
        anomaly_timestamp_list = get_anomaly_timestamp_list(filename=anomaly_filename, machine_id=machine_id)

    with open('MachineData/' + filename, newline='') as csvfile:
        #reader = csv.reader(csvfile, dialect='excel')
        df = pd.read_csv(csvfile, header = None, names=range(121))
        #df.rename(columns={0: 'mach_id'}, inplace=True)

        #Alle Rows entfernen, die nicht zur gewünschten Macshine gehören
        df = df[df[0] == machine_id]

        #Erste vier Spalten entfernen (Maschinen-Id, Zeitstempel, etc.)
        df.drop(labels=range(0, 4), axis='columns', inplace=True)

        #Spalten, die nur einen Wert haben entfernen
        columns_to_exclude = []
        for col in df.columns:
            if len(df[col].unique().tolist()) < 2:
                columns_to_exclude.append(col)
        df.drop(labels=columns_to_exclude, axis='columns', inplace=True)

        for index, row in df.iterrows():

            if anomalies == 'only' and index not in anomaly_timestamp_list:
                continue
            if anomalies == 'exclude' and index in anomaly_timestamp_list:
                continue

            #Werte in float umwandeln
            row.apply(lambda x: float(x))

            csv_list.append(row.values)

        return np.array(csv_list)


def get_anomaly_array(filename, machine_id, sensor_data_count):
    anomaly_timestamp_list = get_anomaly_timestamp_list(filename, machine_id=machine_id)
    anomaly_list = []
    for i in range(0, sensor_data_count):
        if i in anomaly_timestamp_list:
            anomaly_list.append(1)
        else:
            anomaly_list.append(-1)

    return np.array(anomaly_list)


def get_anomaly_timestamp_list(filename, machine_id):
    g = rdflib.Graph()
    g.parse('MachineData/' + filename, format='nt')

    project_hobbit = rdflib.Namespace('http://project-hobbit.eu/resources/debs2017#')
    agt_int = rdflib.Namespace('http://www.agtinternational.com/ontologies/I4.0#')
    agt_int_result = rdflib.Namespace('http://www.agtinternational.com/ontologies/DEBSAnalyticResults#')
    agt_int_weidmuller = rdflib.Namespace('http://www.agtinternational.com/ontologies/WeidmullerMetadata#')
    g.bind('agt_int', agt_int)
    g.bind('agt_int_result', agt_int_result)
    g.bind('agt_int_weidmuller', agt_int_weidmuller)
    g.bind('project_hobbit', project_hobbit)


    anomaly_id_list = []
    anomaly_timestamp_list = []


    for subj, pred, obj in g.triples( (None, agt_int.machine, agt_int_weidmuller['Machine_' + str(machine_id)]) ):
        anomaly_id_list.append(subj)

    for anomaly_id in anomaly_id_list:
        for subj, pred, obj in g.triples( (anomaly_id, agt_int_result.hasTimeStamp, None) ):
            anomaly_timestamp_list.append(int(obj[54:]))


    return anomaly_timestamp_list

