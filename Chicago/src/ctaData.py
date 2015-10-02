# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:59:45 2015

@author: Kevin
"""

import readCTAData
import base36
import numpy as np
from datetime import datetime
import unixDatetime


class ctaBus:
    def __init__(self, filename='../data/bus_route_daily_totals.csv'):
        self.data, self.labels = self.readCTAData.readBusData(filename)
        self.dataByDay, self.dataByDayDates, self.dataByDayRoutes \
            = self.generateDataByDay()

    def filterRoute(self, routeName):
        if type(routeName) != str:
            routeName = str(routeName)
        return self.data[:,0] == base36.base36decode(routeName)

    def generateDataByDay(self):
        routes = np.unique(self.data[:,0])
        dates = np.unique(self.data[:,1])
        dataByDay = np.zeros([len(dates), len(routes)])
        idx = 0
        L = self.data.shape[0]
        for dateIdx, date in enumerate(dates):
            date = dates[dateIdx]
            while idx < L and self.data[idx,1] == date:
                routeNumIdx = [i for i, x in enumerate(routes) if x == self.data[idx,0]]
                dataByDay[dateIdx, routeNumIdx] = self.data[idx,3]
                idx = idx + 1
        return dataByDay, dates, routes
        
        
            
    def readBusData(self,filename='../data/bus_route_daily_totals.csv'):
        num_lines = sum(1 for line in open(filename))
        
        with open(filename) as f:
            line = f.readline().strip()
            labels = line.split(',')
            line = f.readline().strip()
            data = np.zeros([num_lines-1, 4])
            i = 0
            while line:
                line = line.split(',')
                data[i,0] = base36.base36decode(line[0])
                data[i,1] = unixDatetime.ut((datetime.strptime(line[1], '%m/%d/%Y')))
                data[i,2] = base36.base36decode(line[2])
                data[i,3] = int(line[3])
                i = i + 1
                line = f.readline().strip()
            return data, labels


class ctaTrain:
    def __init__(self, filename='../data/L_station_daily_entry_totals.csv'):
        self.data, self.labels = self.readCTAData.readTrainData(filename)
        self.dataByDay, self.dataByDayDates, self.dataByDayRoutes \
            = self.generateDataByDay()

    def filterRoute(self, stationNumber):
        return self.data[:,0] == stationNumber

    def generateDataByDay(self):
            stationNumber = np.unique(self.data[:,0])
            dates = np.unique(self.data[:,1])
            dataByDay = np.zeros([len(dates), len(stationNumber)])
            idx = 0
            L = self.data.shape[0]
            for dateIdx, date in enumerate(dates):
                date = dates[dateIdx]
                while idx < L and self.data[idx,1] == date:
                    stationNumberIdx = [i for i, x in enumerate(stationNumber) if x == self.data[idx,0]]
                    dataByDay[dateIdx, stationNumberIdx] = self.data[idx,3]
                    idx = idx + 1
            return dataByDay, dates, stationNumber
    
    def readTrainData(self,filename='../data/L_station_daily_entry_totals.csv'):
        num_lines = sum(1 for line in open(filename))
        
        with open(filename) as f:
            line = f.readline().strip()
            labels = line.split(',')
            line = f.readline().strip()
            data = np.zeros([num_lines-1, 4])
            i = 0
            while line:
                line = line.split(',')
                data[i,0] = int(line[0])
                data[i,1] = unixDatetime.ut(datetime.strptime(line[2], '%m/%d/%Y'))
                data[i,2] = base36.base36decode(line[3])
                data[i,3] = int(line[4])
                i = i + 1
                line = f.readline().strip()
            return data, labels
        
