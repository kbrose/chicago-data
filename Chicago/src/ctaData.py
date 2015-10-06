# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:59:45 2015

@author: Kevin
"""

import base36
import numpy as np
from datetime import datetime
import unixDatetime
import matplotlib.pyplot as plt
import csaps


class bus:
    def __init__(self, filename='../data/bus_route_daily_totals.csv'):
        self.data, self.labels = self.readBusData(filename)
        self.dataByDay, self.dataByDayDates, self.dataByDayRoutes \
            = self.generateDataByDay()
    
    def plotRoutes(self, routes, p=1, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        dates = map(lambda x: unixDatetime.dt(x), self.dataByDayDates)
        date_ords = map(lambda x: x.toordinal(), dates)

        legend_labels = []
        lines = []
        for route in routes:
            ax.hold(True);
            if type(route) is not str:
                route = str(route)
            route = base36.base36decode(route)
            filt = self.dataByDayRoutes == route
            if not filt.any():
                print('Route ' + base36.base36encode(route) + ' appears to not exist.')
                continue
            legend_labels = legend_labels + [base36.base36encode(route)]
            line = plt.plot(dates,
                            csaps.csaps(date_ords, self.dataByDay[:,filt], p, date_ords),
                            axes=ax)
            lines = lines + [line]
        plt.legend(legend_labels);
        return ax, lines

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
            while idx < L and self.data[idx,1] == date:
                routeNumIdx = np.nonzero(routes == self.data[idx,0])[0]
                dataByDay[dateIdx, routeNumIdx] = self.data[idx,3]
                idx = idx + 1
        return dataByDay, dates, routes

    def readBusData(self,filename='../data/bus_route_daily_totals.csv'):
        num_lines = sum(1 for line in open(filename))
        
        with open(filename) as f:
            line = f.readline().strip()
            labels = line.split(',')
            line = f.readline().strip().split(',')
            data = np.zeros([num_lines-1, 4])
            i = 0
            while line[0] != '':
                data[i,0] = base36.base36decode(line[0])
                data[i,1] = unixDatetime.ut((datetime.strptime(line[1], '%m/%d/%Y')))
                data[i,2] = base36.base36decode(line[2])
                data[i,3] = int(line[3])
                i = i + 1
                line = f.readline().strip().split(',')
            return data, labels


class train:
    def __init__(self, filename='../data/L_station_daily_entry_totals.csv'):
        self.data, self.labels = self.readTrainData(filename)
        self.dataByDay, self.dataByDayDates, self.dataByDayStops \
            = self.generateDataByDay()

    def filterRoute(self, stationNumber):
        return self.data[:,0] == stationNumber

    def generateDataByDay(self):
            stationNumbers = np.unique(self.data[:,0])
            dates = np.unique(self.data[:,1])
            dataByDay = np.zeros([len(dates), len(stationNumbers)])
            idx = 0
            L = self.data.shape[0]
            for dateIdx, date in enumerate(dates):
                while idx < L and self.data[idx,1] == date:
                    stationNumberIdx = np.nonzero(stationNumbers == self.data[idx,0])[0]
                    dataByDay[dateIdx, stationNumberIdx] = self.data[idx,3]
                    idx = idx + 1
            return dataByDay, dates, stationNumbers

    def readTrainData(self,filename='../data/L_station_daily_entry_totals.csv'):
        num_lines = sum(1 for line in open(filename))

        with open(filename) as f:
            line = f.readline().strip()
            labels = line.split(',')
            line = f.readline().strip().split(',')
            data = np.zeros([num_lines-1, 4])
            i = 0
            while line[0] != '':
                data[i,0] = int(line[0])
                data[i,1] = unixDatetime.ut(datetime.strptime(line[2], '%m/%d/%Y'))
                data[i,2] = base36.base36decode(line[3])
                data[i,3] = int(line[4])
                i = i + 1
                line = f.readline().strip().split(',')
            return data, labels
        
