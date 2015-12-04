# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:59:45 2015

@author: Kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pykml.parser

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

        Notes
        -----
        Matplotlib stacked area charts do not play well with NaN values.
        If one of the specified routes is NaN at a given coordinate, then
        all routes listed *after* that route in the function call will not
        be plotted. Thus, if you have set stacked=True then it is
        recommended that you also set fillzero=True.
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

        if not routes: # no routes to plot
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
        data = self.data.fillna(value=0, inplace=False)

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

    def plot_route_shapes(self, routes=None, transparency=True):
        '''
        Plots the shapes of the specified routes using lat/long
        pairs. The routes can have their transparency set based
        on how many people have ridden the bus since the data
        starts in 2001.

        Parameters
        ----------
        routes       : List of routes to plot. The default is to plot
                       all possible routes. Note that some routes are
                       in the daily ridership data but NOT in the
                       route shape data. This can happen because the
                       route is older than the shape data, or because
                       it is too small.
        transparency : Boolean. Set to True if you want the routes
                       to have variable transparency as described
                       above. Set to False to make all routes opaque.

        Returns
        -------
        ax : matplotlib axes object containing the plot.

        Notes
        -----
        The axis is set to have a square aspect ratio in order to
        preserve the shape of the routes. Use ax.set_aspect('auto')
        to set it back to the default value.
        '''
        plt.figure()
        ax = plt.gca()

        shapes = self.route_shapes()

        if routes is None:
            routes = self.routes()

        if type(routes) is not list:
            routes = [routes]

        max_ridership = self.data.fillna(0).mean().max()

        for idx, route in enumerate(routes):
            r = str(route).upper()
            if r not in shapes.keys():
                print('route ' + r + ' was not found in the shapefile.')
                continue

            if transparency:
                alpha = self.data[r].fillna(0).mean()
                alpha = alpha / max_ridership
            else:
                alpha = 1

            line_segs = shapes[r]
            cond_shape = self.condense_shape(line_segs)
            for shape in cond_shape:
                ax.plot(shape[:,1], shape[:,0], alpha=alpha, color=(.8,.9,1), linewidth=1)

        ax.set_aspect('equal', 'datalim')
        ax.set_axis_bgcolor((0,0,0))
        return ax

    def routes(self):
        """
        Returns a list of routes as strings.
        """
        return self.data.columns.tolist()[1:] # [1:] gets rid of "daytype" col

    @staticmethod
    def condense_shape(shape):
        '''
        Takes in a list of line segments and condenses them by joining
        consecutive line segments using a greedy algorithm.

        Parameters
        ----------
        shape : Nx2x2 numpy array corresponding to line segments. For example,
                one of the values in the dictionary returned by route_shapes.

        Returns
        -------
        cond_shape : List of Mx2 arrays where line segments have been
                     joined together if they share a coordinate.

        Notes
        -----
        This method employs a simple greedy algorithm to build up the lists.
        There is no guarantee that the number of elements in the returned
        list is optimally low.
        '''
        cond_shape = []
        curr_path = shape[0].tolist()
        shape = shape[1:]
        while shape.shape[0]:
            try:
                share_coords = map(lambda x: all(x), shape[:,0,:] == curr_path[0]).index(True)
                curr_path = [shape[share_coords,1].tolist()] + curr_path
                shape = np.vstack((shape[0:share_coords], shape[share_coords+1:]))
                continue
            except ValueError:
                pass
            try:
                share_coords = map(lambda x: all(x), shape[:,1,:] == curr_path[0]).index(True)
                curr_path = [shape[share_coords,0].tolist()] + curr_path
                shape = np.vstack((shape[0:share_coords], shape[share_coords+1:]))
                continue
            except ValueError:
                pass
            try:
                share_coords = map(lambda x: all(x), shape[:,0,:] == curr_path[-1]).index(True)
                curr_path = curr_path + [shape[share_coords,1].tolist()]
                shape = np.vstack((shape[0:share_coords], shape[share_coords+1:]))
                continue
            except ValueError:
                pass
            try:
                share_coords = map(lambda x: all(x), shape[:,1,:] == curr_path[-1]).index(True)
                curr_path = curr_path + [shape[share_coords,0].tolist()]
                shape = np.vstack((shape[0:share_coords], shape[share_coords+1:]))
                continue
            except ValueError:
                pass
            cond_shape.append(curr_path)
            curr_path = shape[0].tolist()
            shape = shape[1:]

        return map(lambda x: np.array(x), cond_shape)

    def route_shapes(self, filename=None):
        '''
        Returns a set of shapes (i.e. list of (lat,lon) coordinate pairs)
        describing the bus routes. The expected input is a KML file with a
        similar structure as ../data/CTABusRoutes.kml

        Parameters
        ----------
        filename : path to the .kml file that should be parsed.
                   DEFAULT: ../data/CTABusRoutes.kml

        Returns
        -------
        route_coords : A dictionary of shapes for bus routes. The keys
                       are the bus route names and the values are Nx2x2
                       numpy arrays. If X is one of these arrays, then
                       X[i] describes a line segment such that X[i,0,:]
                       and X[i,1,:] are the two ends of the line segment.
                       The line segments are "dictionary sorted", i.e.
                       sorted along their first coordinate and then their
                       second coordinate.
        '''
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__),
                                    '../data/CTABusRoutes.kml')

        with open(filename, 'r') as f:
            root = pykml.parser.fromstring(f.read())

        route_coords = {}
        for route in root.Document.Folder.iterchildren():
            if 'Placemark' in route.tag:
                name = str(route.name)
                line_strings = route.MultiGeometry.getchildren()
                coords = [l.coordinates.text for l in line_strings]

                # Convert string to numpy array
                coords = map(lambda x: self.__parse_coords(x), coords)

                # Change from lists of points to lists of line segments
                # i.e., pairs of consecutive points
                for i, arr in enumerate(coords):
                    coords[i] = [np.array(arr[j:j+2,:]) for j in range(len(arr)-1)]

                # Flatten the list of line segments
                coords = np.array([item for sublist in coords for item in sublist])

                # do a dictionary sort on the coordinates
                # facilitates removal of duplicates later.
                for i in range(len(coords)):
                    if coords[i][0,0] < coords[i][1,0]:
                        continue
                    if coords[i][0,0] > coords[i][1,0] or coords[i][0,1] > coords[i][1,1]:
                        coords[i] = np.flipud(coords[i])

                route_coords[name] = coords

        return route_coords

    @staticmethod
    def __parse_coords(coords_text):
        coords = coords_text.split(' ')[1:]
        coords = map(lambda c: c.split(',')[0:2], coords)
        coords = map(lambda xy_pair: [float(c) for c in xy_pair[::-1]], coords)
        return np.array(coords)

    @staticmethod
    def __read_bus_data(filename):
        data = pd.read_csv(filename)
        data['date'] = utils.lookup(data['date'])
        daytypes = data.drop_duplicates(subset='date')['daytype']
        data = data.pivot(index='date', columns='route', values='rides')
        data.insert(0,0,daytypes)
        data.rename(columns={0:'daytype'}, inplace=True)
        return data

    def __str__(self):
        return 'cta bus daily ridership data for routes [' + ', '.join(self.routes()) + '].'


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

    @staticmethod
    def __read_train_data(filename):
        data = pd.read_csv(filename)
        data['date'] = utils.lookup(data['date'])

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

    @staticmethod
    def __get_L_stop_names(filename):
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

class utils:
    """
    Class containing all utility functions used by transit classes.
    """

    @staticmethod
    def lookup(s):
        """
        This is an extremely fast approach to datetime parsing.
        For large data, the same dates are often repeated. Rather than
        re-parse these, we store all unique dates, parse them, and
        use a lookup to convert all dates.

        Thanks to fixxxer, found at
        http://stackoverflow.com/questions/29882573
        """
        dates = {date:pd.to_datetime(date) for date in s.unique()}
        return s.apply(lambda v: dates[v])
