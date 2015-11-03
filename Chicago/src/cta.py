# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:59:45 2015

@author: Kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class bus:
    """
    Class with functions and data related to Chicago Transit Authority (CTA)
    data pertaining to bus rides broken down by day and route.
    """

    def __init__(self, filename=None):
        """
        Loads the data from the CSV file. If filename is not given then
        ../data/bus_route_daily_totals.csv is used.
        """
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__),
                                    '../data/bus_route_daily_totals.csv')
        self.data = self.__read_bus_data(filename)

    def plot_routes(self, routes, resamp=None, fillzero=False, stacked=False, ax=None):
        """
        Plots all specified routes on the same axis.

        Parameters
        ----------
        routes   : list of routes, can be integers or strings, i.e. [2, 6, 'x28']
        resamp   : String representing how to downsample for plotting. Use 'W' for
                   weekly, 'M' for monthly, 'Q' for quarterly, or 'AS' for yearly.
                   DEFAULT: resamp = None # no resampling
                   http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        fillzero : Set to True to fill in missing ridership data with 0. Data is
                   considered missing for any day the bus was not running.
                   This can include weekends for commuter buses, or times when
                   the bus route did not exist.
                   DEFAULT: fillzero = False
        stacked  : True to produce a stacked area plot showing total amounts as
                   well as individual route contributions.
        ax       : axes object to plot to.
                   DEFAULT: ax = None

        Returns
        -------
        ax : axes object the object was plotted on
        """
        # create new figure if no axis provided
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # be flexible with "scalar" inputs
        if not isinstance(routes, (list, np.ndarray)):
            routes = [routes]

        # resample and fill missing with 0 as necessary
        data = self.data.copy()
        if fillzero:
            data.fillna(value=0, inplace=True)
        if resamp is not None:
            data = data.resample(resamp) # mean resampling is default

        # Convert routes to strings if they are not already
        routes = map(lambda x: str(x).upper(), routes)

        # make sure route is valid
        for i in xrange(len(routes)):
            route = routes[i]
            if route not in data.columns:
                print('Invalid route ' + route + ', check *.rotues() for complete list')
                del(routes[routes.index(route)])

        if not routes:
            return ax

        # do this awful mess to deal with stacked plot legends not showing up
        if stacked:
            poly_collections = ax.stackplot(data.index, data[routes].T, linewidth=0)
            legend_proxies = []
            for i, poly in enumerate(poly_collections):
                # poly.set_gid(legend_labels[i])
                legend_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))
            ax.legend(legend_proxies, routes, ncol=max([(len(routes)/30), 1]),
                      bbox_to_anchor=[1, .5], loc='center left')
        else:
            data[routes].plot(ax=ax)

        return ax

    def plot_fft(self, routes, ax=None):
        """
        Plot the [discrete/fast] fourier transform of bus riderships per day
        for all the specified routes on the optionally specified axis. The FFT
        is normalized to sum to 1 so that different routes can be compared
        more meaningfully.

        Parameters
        ----------
        routes : list of routes, can be integers or strings, i.e. [2, 6, 'x28']
                 routes can also be a scalar, i.e. just "x28" or 2
        ax     : axes object to plot to.
                 DEFAULT: ax = None

        Returns
        -------
        ax    : axes object the object was plotted on
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        hold = ax._hold
        ax.hold(True)

        if not isinstance(routes, (list, np.ndarray)):
            routes = [routes]

        # fill missing with 0 as necessary
        data = self.data.copy()
        data.fillna(value=0, inplace=True)

        # Convert routes to strings if they are not already
        routes = map(lambda x: str(x).upper(), routes)

        legend_labels = []
        for route in routes:

            time_x = data[route]
            y = np.absolute(np.fft.rfft(time_x))
            y = y / np.sum(y)
            x = np.fft.rfftfreq(time_x.size, 1)

            legend_labels.append(route)
            plt.plot(x, y, axes=ax, gid=route)

        plt.legend(legend_labels)
        ax.hold(hold)

        return ax

    def routes(self):
        """
        Returns a list of routes as strings.
        """
        return self.data.columns.tolist()[1:] # gets rid of "daytype" col

    def __read_bus_data(self, filename):
        data = pd.read_csv(filename,parse_dates=[1])
        daytypes = data.drop_duplicates(subset='date')['daytype']
        data = data.pivot(index='date', columns='route', values='rides')
        data.insert(0,0,daytypes)
        data.rename(columns={0:'daytype'}, inplace=True)
        return data

class train:
    """
    Class with functions and data related to Chicago Transit Authority (CTA)
    data pertaining to train rides broken down by day and station.
    """

    def __init__(self, rides_filename=None, station_filename=None):
        """
        Loads the data from CSV files.
        """
        if rides_filename is None:
            rides_filename = os.path.join(os.path.dirname(__file__),
                                          '../data/L_station_daily_entry_totals.csv')
        self.data = self.__read_train_data(rides_filename)

        if station_filename is None:
            station_filename = os.path.join(os.path.dirname(__file__),
                                            '../data/L_Stops.csv')
        self.stop_data = self.__get_L_stop_names(station_filename)

    def plot_stops(self, stops, resamp=None, fillzero=False, stacked=False, ax=None):
        """
        Plots all specified routes on the same axis.

        Parameters
        ----------
        stops    : list of stops, can be either integers corresponding to the
                   station ID (denoted MAP_ID in self.stop_data) or a string
                   corresponding to the station name. The station name will
                   attempt to be matched to the STOP_DESCRIPTIVE_NAME, STATION_NAME,
                   and STOP_NAME in that order (again, these labels are found
                   in self.stop_data).
                   See parse_stop for more information
        resamp   : String representing how to downsample for plotting. Use 'W' for
                   weekly, 'M' for monthly, 'Q' for quarterly, or 'AS' for yearly.
                   http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
                   DEFAULT: resamp = None # no resampling
        fillzero : Set to True to fill in missing ridership data with 0. Data is
                   considered missing for any day the train station was not open.
                   This can include weekends for commuter stations, or times when
                   the station did not exist.
                   DEFAULT: fillzero = False
        stacked  : True to produce a stacked area plot showing total amounts as
                   well as individual route contributions.
        ax       : axes object to plot to.
                   DEFAULT: ax = None

        Returns
        -------
        ax : axes object the object was plotted on
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not isinstance(stops, (list, np.ndarray)):
            stops = [stops]

        # resample and fill missing with 0 as necessary
        data = self.data.copy()
        if fillzero:
            data.fillna(value=0, inplace=True)
        if resamp is not None:
            data = data.resample(resamp) # mean resampling is default

        # Convert routes to strings if they are not already
        stops = map(lambda x: str(x).upper(), stops)

        # make sure stop is valid
        for i in xrange(len(stops)):
            stop = stops[i]
            stop_col_name = self.parse_stop(stop)
            if stop_col_name:
                stops[i] = stop_col_name
            else:
                print('Invalid stop ' + stop + ', check *.stops() for complete list')
                del(stops[stops.index(stop)])

        if not stops:
            return ax

        # do this awful mess to deal with stacked plot legends not showing up
        if stacked:
            poly_collections = ax.stackplot(data.index, data[stops].T, linewidth=0)
            legend_proxies = []
            for i, poly in enumerate(poly_collections):
                # poly.set_gid(legend_labels[i])
                legend_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))
            ax.legend(legend_proxies, stops, ncol=max([(len(stops)/30), 1]),
                      bbox_to_anchor=[1, .5], loc='center left')
        else:
            data[stops].plot(ax=ax)

        return ax

    def plot_fft(self, stops, ax=None):
        """
        Plot the [discrete/fast] fourier transform of train riderships per day
        for all the specified stops on the optionally specified axes object. The
        FFT is normalized so that it sums to 1. This allows different stops to be
        compared more meaningfully.

        Parameters
        ----------
        stops : list of stops, can be integers or strings, i.e. [2, 6, 'x28']
                stops can also be a scalar, i.e. just "x28" or 2
        ax    : axes object to plot to.
                DEFAULT: ax = None

        Returns
        -------
        ax    : axes object the object was plotted on
        lines : list of line objects plotted for each route.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        hold = ax._hold
        ax.hold(True)

        if not isinstance(stops, (list, np.ndarray)):
            stops = [stops]

        # fill missing with 0 as necessary
        data = self.data.copy()
        data.fillna(value=0, inplace=True)

        legend_labels = []
        for s in stops:
            stop = self.parse_stop(s)
            if not stop:
                print('Invalid stop ' + str(s) + ', check *.stops() for complete list')
                continue
            time_x = data[stop]
            y = np.absolute(np.fft.rfft(time_x))
            y = y / np.sum(y)
            x = np.fft.rfftfreq(time_x.size, 1)

            legend_labels.append(stop)
            plt.plot(x, y, axes=ax, gid=stop)

        plt.legend(legend_labels)
        ax.hold(hold)

        return ax

    def get_stop_by_color(self, colors):
        """
        Get a list of stations that have all the specified
        colors running through it.

        Parameters
        ----------
        colors : List of strings, i.e. 'RED', 'PURPLE'.

        Returns
        -------
        stops : List of stops that have every specified color
        """
        if type(colors) is not list:
            colors = [colors]
        colors = map(lambda x: x.upper(), colors)
        try:
            grouped = self.stop_data[['MAP_ID', 'STATION_NAME'] + colors].groupby('MAP_ID')
            grouped = grouped.max() # maximum is basically an "or" of the group
            filt = grouped[colors.pop()]
            for c in colors:
                filt = filt & grouped[c]
            data = grouped[filt].reset_index(level=0)
        except KeyError:
            print('Unexpected color, try *.stop_data.columns[7:-1] for complete list')
            return None
        return data['MAP_ID'].map(lambda x: self.parse_stop(x)).tolist()


    def parse_stop(self, stop):
        """
        Parses a stop ID or name into a standard format.

        Parameters
        ----------
        stop : list of stops, can be either integers corresponding to the
               station ID (denoted MAP_ID in self.stop_data) or
               a string corresponding to the station name. The station name will
               attempt to be matched to the STOP_DESCRIPTIVE_NAME, STATION_NAME, and
               STOP_NAME in that order (again, these labels are found
               in self.stop_data).

        Returns
        -------
        String in the format "MAP_ID: STATION_NAME" matching the column name in self.data
        """

        data = self.stop_data.copy()
        cols = self.data.columns[1:]
        col_num = map(lambda x: int(x[0:5]), cols)

        try:
            if type(stop) is str:
                if len(stop) > 5:
                    stop_num = int(stop[0:5])
                else:
                    stop_num = int(stop)
            else:
                stop_num = stop
            map_id_equal = data['MAP_ID'] == stop_num
            if any(map_id_equal):
                occurence = data[map_id_equal].iloc[0]
                return cols[col_num.index(occurence['MAP_ID'])]
        except (TypeError, ValueError):
            pass

        data['lower_case_column'] = data['STATION_DESCRIPTIVE_NAME'].map(lambda x: x.lower())
        try:
            stop_descriptive_name_equal = data['lower_case_column'] == stop.lower()
            if any(stop_descriptive_name_equal):
                occurence = data[stop_descriptive_name_equal].iloc[0]
                return cols[col_num.index(occurence['MAP_ID'])]
        except (TypeError, AttributeError, ValueError):
            pass

        data['lower_case_column'] = data['STATION_NAME'].map(lambda x: x.lower())
        try:
            station_name_equal = data['lower_case_column'] == stop.lower()
            if any(station_name_equal):
                occurence = data[station_name_equal].iloc[0]
                return cols[col_num.index(occurence['MAP_ID'])]
        except (TypeError, AttributeError, ValueError):
            pass

        data['lower_case_column'] = data['STOP_NAME'].map(lambda x: x.lower())
        try:
            stop_name_equal = data['lower_case_column'] == stop.lower()
            if any(stop_name_equal):
                occurence = data[stop_name_equal].iloc[0]
                return cols[col_num.index(occurence['MAP_ID'])]
        except (TypeError, AttributeError, ValueError):
            pass

        return False

    def stops(self):
        """
        Returns a list of all of the available stops.
        """
        return self.data.columns.tolist()[1:]

    def __read_train_data(self, filename):
        data = pd.read_csv(filename, parse_dates=[2])

        # Combine the ID and station-name fields into one column
        ids = data['station_id'].map(lambda x: str(x))
        names = data['station_name']
        id_names = ids + ': ' + names
        data.drop(['station_id', 'station_name'], inplace=True, axis=1)
        data['station_id_name'] = id_names

        # Save the daytypes
        daytypes = data.drop_duplicates(subset='date')['daytype']

        # Pivot, insert the daytypes back in
        data = data.pivot_table(index='date', columns='station_id_name', values='rides')
        data.insert(0,0,daytypes)
        data.rename(columns={0:'daytype'}, inplace=True)
        return data


    def __get_L_stop_names(self, filename):
        data = pd.read_csv(filename)

        # Convert this from string "(x,y)" to actual tuple
        data['Location'] = data['Location'].map(lambda x:
            (float(x[1:-1].split(',')[0]), float(x[1:-1].split(',')[1])))

        # Rename those awful default names
        data.rename(columns={'G':'GREEN', 'BRN':'BROWN', 'P':'PURPLE',
                             'Pexp':'PINK_EXPRESS', 'Y':'YELLOW',
                             'Pnk':'PINK', 'O':'ORANGE'},
                    inplace=True)

        return data

