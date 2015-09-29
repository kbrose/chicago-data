# -*- coding: utf-8 -*-

import numpy as np
import base36
from datetime import datetime
import unixDatetime

def readBusData():
    num_lines = sum(1 for line in open('../data/bus_route_daily_totals.csv'))
    
    with open('../data/bus_route_daily_totals.csv') as f:
        line = f.readline()
        labels = line.split(',')
        line = f.readline()
        data = np.zeros([num_lines, 4])
        i = 0
        while line:
            line = line.split(',')
            data[i,0] = base36.base36decode(line[0])
            data[i,1] = unixDatetime.ut((datetime.strptime(line[1], '%m/%d/%Y')))
            data[i,2] = base36.base36decode(line[2])
            data[i,3] = int(line[3])
            i = i + 1
            line = f.readline()
        return data, labels

def readTrainData():
    num_lines = sum(1 for line in open('../data/L_station_daily_entry_totals.csv'))
    
    with open('../data/L_station_daily_entry_totals.csv') as f:
        line = f.readline()
        labels = line.split(',')
        line = f.readline()
        data = np.zeros([num_lines, 4])
        i = 0
        while line:
            line = line.split(',')
            data[i,0] = base36.base36decode(line[0])
            data[i,1] = unixDatetime.ut(datetime.strptime(line[2], '%m/%d/%Y'))
            data[i,2] = base36.base36decode(line[3])
            data[i,3] = int(line[4])
            i = i + 1
            line = f.readline()
        return data, labels
        
