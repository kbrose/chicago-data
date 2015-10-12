# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:59:45 2015

@author: Kevin
"""

import base36
import numpy as np
from datetime import datetime
import unix_datetime as udt
import matplotlib.pyplot as plt
import csaps
import re
import scipy.stats


class bus:
    """
    Class with functions and data related to Chicago Transit Authority (CTA)
    data pertaining to bus rides broken down by day and route.
    """

    def __init__(self, filename='../data/bus_route_daily_totals.csv'):
        """
        Loads the data from the CSV files
        """
        raw_data, raw_labels = self.__read_bus_data(filename)
        self.data, self.dates, self.routes = self.__generate_data_by_day(raw_data, raw_labels)

    def plot_routes(self, routes, p=1, stacked=False, ax=None):
        """
        Plots all specified routes on the same axis.

        Parameters
        ----------
        routes : list of routes, can be integers or strings, i.e. [2, 6, 'x28']
        p      : smoothing parameter, 0 <= p <= 1, lower values result in more
                 smoothing. See csaps.
                 DEFAULT: p = 1 for perfect interpolation
        ax     : axes object to plot to.
                 DEFAULT: ax = None
        stacked: True to produce a stacked area plot showing total amounts as well
                 as individual route contributions.

        Returns
        -------
        ax : axes object the object was plotted on
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not isinstance(routes, (list, np.ndarray)):
            routes = [routes]

        dates = map(lambda x: udt.dt(x), self.dates)
        date_ords = map(lambda x: x.toordinal(), dates)

        legend_labels = []
        if stacked:
            y = np.tile(np.nan, (len(dates), len(routes)))
            stack_idx = 0
        for route in routes:
            ax.hold(True)
            if type(route) is str:
                route_name = route
                route = base36.tobase10(route)
            else:
                route_name = str(route)
                route = base36.tobase10(str(int(route)))

            filt = self.routes == route
            if not filt.any():
                print('Route ' + route_name + ' appears to not exist.')
                continue

            legend_labels.append(route_name.upper())
            smoothed_data = csaps.csaps(date_ords, self.data[:, filt], p, date_ords)
            smoothed_data[smoothed_data < 0] = 0

            if stacked:
                y[:,stack_idx] = smoothed_data
                stack_idx += 1
            else:
                ax.plot(dates, smoothed_data, gid=route_name)

        if stacked:
            poly_collections = ax.stackplot(dates, y.T, baseline='zero', linewidth=0)
            legend_proxies = []
            for i, poly in enumerate(poly_collections):
                poly.set_gid(legend_labels[i])
                legend_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))
            ax.legend(legend_proxies, legend_labels, ncol=max([(len(routes)/30), 1]),
                      bbox_to_anchor=[1, .5], loc='center left')
        else:
            ax.legend(legend_labels, ncol=max([(len(routes)/30), 1]),
                      bbox_to_anchor=[1, .5], loc='center left')

        return ax

    def plot_fft(self, routes, ax=None):
        """
        Plot the [discrete/fast] fourier transform of bus riderships per day
        for all the specified routes on the optionally specified axis. The FFT
        is normalized so that it sums to 1 so that different routes can be
        compared more meaningfully.

        Parameters
        ----------
        routes : list of routes, can be integers or strings, i.e. [2, 6, 'x28']
                 routes can also be a scalar, i.e. just "x28" or 2
        ax     : axes object to plot to.
                 DEFAULT: ax = None

        Returns
        -------
        ax    : axes object the object was plotted on
        lines : list of line objects plotted for each route.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not isinstance(routes, (list, np.ndarray)):
            routes = [routes]

        legend_labels = []
        lines = []
        for route in routes:
            ax.hold(True)
            if type(route) is not str:
                route = base36.tobase36(route)
            route = base36.tobase10(route)

            filt = self.routes == route
            route_name = base36.tobase36(route)

            if not filt.any():
                print('Route ' + route_name + ' appears to not exist.')
                continue

            time_x = self.data[:,filt].squeeze()
            y = np.absolute(np.fft.rfft(time_x))
            y = y / np.sum(y)
            x = np.fft.rfftfreq(time_x.size, 1)

            legend_labels.append(route_name)
            line = plt.plot(x, y, axes=ax, gid=route_name)
            lines = lines + line

        plt.legend(legend_labels)
        ax.hold(False) # TODO: keep the setting that it had...

        return ax, lines

    def detect_change(self, route):
        """
        Returns a list of dates when the specified route underwent
        a "significant service change". A significant service change
        is defined as the non-zero riderships of the next 12 months
        being statistically significantly different from the previous
        12 months. Note that this does not capture a change in a bus
        that goes from running every day to only week-days, but
        keeping a relatively constant amount on week-days.

        We only test for difference at the start of a month, and if
        a year is flagged as different then every other month in that
        year is not tested for another change.
        """
        if type(route) is str:
            route = base36.tobase10(route)
        else:
            route = base36.tobase10(str(int(route)))
        route_idx = np.nonzero(self.routes == route)[0]

        changes = []
        d_changes = []
        p_vals = []
        d_p_vals = []

        win = 365

        for i, d in enumerate(self.dates):
            if i < win or i >= len(self.dates) - win:
                continue
            if udt.dt(d).day == 1:
                prev_yr = self.data[i-win+1:i+1, route_idx]
                curr_yr = self.data[i:i+win, route_idx]
                # TODO: explain why plus 1

                p = scipy.stats.ks_2samp(curr_yr.flatten(), prev_yr.flatten())[1]
                p_vals.append(p)
                d_p_vals.append(udt.dt(d))
                if p < 1e-6:
                    changes.append(p)
                    d_changes.append(udt.dt(d))
                #print(str(udt.dt(d)) + ' : ' + str(md))
        changes = np.array(changes)
        loc_mins = np.r_[True, changes[1:] < changes[:-1]] & np.r_[changes[:-1] < changes[1:], True]
        d_changes = np.array(d_changes)[loc_mins].tolist()
        changes = changes[loc_mins].tolist()

        return d_changes, changes, d_p_vals, p_vals

    def show_changes(self, route):
        """
        Plot where changes have been detected
        """
        d_changes, changes, d_p_vals, p_vals = self.detect_change(route)

        fig, axes = plt.subplots(2, 1, sharex=True)

        self.plot_routes(route, .01, ax=axes[0])
        color = next(axes[0]._get_lines.color_cycle)
        for d in d_changes:
            yl = axes[0].get_ylim()
            axes[0].plot([d, d], yl, color=color)
        axes[0].legend(['bus', 'changes'])

        # TODO: plot the cut-off p-value that I'm using
        axes[1].semilogy(d_p_vals, p_vals)

    def routes_to_base36(self):
        """
        Returns the routes in base 36, i.e. the normal, human-readable
        format. This is useful for plotting all routes:

        >>> import cta
        >>> bus = cta.bus()
        >>> bus.plot_routes(bus.routes_to_base36(), .01)
        >>> bus.plot_fft(bus.routes_to_base36())
        """
        return map(lambda x: base36.tobase36(x), self.routes)

    def __generate_data_by_day(self, raw_data, raw_labels):
        routes = np.unique(raw_data[:, 0])
        dates = np.unique(raw_data[:, 1])
        dataByDay = np.zeros([len(dates), len(routes)])
        idx = 0
        L = raw_data.shape[0]
        for dateIdx, date in enumerate(dates):
            while idx < L and raw_data[idx, 1] == date:
                routeNumIdx = np.nonzero(routes == raw_data[idx, 0])[0]
                dataByDay[dateIdx, routeNumIdx] = raw_data[idx, 3]
                idx = idx + 1
        return dataByDay, dates, routes

    def __read_bus_data(self, filename='../data/bus_route_daily_totals.csv'):
        num_lines = sum(1 for line in open(filename))

        with open(filename) as f:
            line = f.readline().strip()
            labels = line.split(',')
            line = f.readline().strip().split(',')
            data = np.zeros([num_lines-1, 4])
            i = 0
            while line[0] != '':
                data[i, 0] = base36.tobase10(line[0])
                data[i, 1] = udt.ut((datetime.strptime(line[1], '%m/%d/%Y')))
                data[i, 2] = base36.tobase10(line[2])
                data[i, 3] = int(line[3])
                i = i + 1
                line = f.readline().strip().split(',')
            return data, labels


class train:
    """
    Class with functions and data related to Chicago Transit Authority (CTA)
    data pertaining to train rides broken down by day and station.
    """

    def __init__(self, filename='../data/L_station_daily_entry_totals.csv'):
        """
        Loads the data from CSV files.
        """
        raw_data, raw_labels = self.__read_train_data(filename)
        self.data, self.dates, self.stops = self.__generate_data_by_day(raw_data, raw_labels)
        self.stop_data, self.stop_data_labels = self.__get_L_stop_names()

    def plot_stops(self, stops, p=1, ax=None):
        """
        Plots all specified routes on the same axis.

        Parameters
        ----------
        stops  : list of stops, can be either integers corresponding to the
                 station ID (denoted MAP_ID in self.stop_data_labels) or
                 a string corresponding to the station name. The station name will
                 attempt to be matched to the STOP_DESCRIPTIVE_NAME, STATION_NAME, and
                 STOP_NAME in that order (again, these labels are found
                 in self.stop_data_labels).
        p      : smoothing parameter, 0 <= p <= 1, lower values result in more
                 smoothing. See csaps.
                 DEFAULT: p = 1 for perfect interpolation
        ax     : axes object to plot to.
                 DEFAULT: ax = None

        Returns
        -------
        ax    : axes object the object was plotted on
        lines : list of line objects for each stop.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not isinstance(stops, (list, np.ndarray)):
            stops = [stops]

        dates     = map(lambda x: udt.dt(x), self.dates)
        date_ords = map(lambda x: x.toordinal(), dates)

        station_names          = map(lambda x: x[3].lower(), self.stop_data)
        stop_names             = map(lambda x: x[2].lower(), self.stop_data)
        stop_descriptive_names = map(lambda x: x[4].lower(), self.stop_data)
        map_ids                = map(lambda x: x[5], self.stop_data)

        legend_labels = []
        lines = []
        for stop in stops:
            ax.hold(True)

            if type(stop) is str:
                stopl = stop.lower()
                map_id = [map_ids[i] for i, x in enumerate(stop_descriptive_names) if x == stopl]
                if not map_id:
                    map_id = [map_ids[i] for i, x in enumerate(stop_names) if x == stopl]
                    if not map_id:
                        map_id = [map_ids[i] for i, x in enumerate(station_names) if x == stopl]
                        if not map_id:
                            print('Stop "' + stop + '" cannot be found.')
                            continue
                map_id = map_id[0]
                filt = self.stops == map_id
            else:
                filt = self.stops == stop
                map_id = stop

            if not filt.any():
                print('Stop ' + str(stop) + ' cannot be found.')
                continue

            station_name = self.station_name(map_id, 1)

            legend_labels.append(station_name)
            smoothed_data = csaps.csaps(date_ords, self.data[:, filt].sum(axis=1), p, date_ords)
            smoothed_data[smoothed_data < 0] = 0
            line = plt.plot(dates, smoothed_data,
                            axes=ax, gid=station_name)
            lines = lines + line

        plt.legend(legend_labels)
        return ax, lines

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

        if not isinstance(stops, (list, np.ndarray)):
            stops = [stops]

        station_names          = map(lambda x: x[3].lower(), self.stop_data)
        stop_names             = map(lambda x: x[2].lower(), self.stop_data)
        stop_descriptive_names = map(lambda x: x[4].lower(), self.stop_data)
        map_ids                = map(lambda x: x[5], self.stop_data)

        legend_labels = []
        lines = []
        for stop in stops:
            ax.hold(True)

            if type(stop) is str:
                stopl = stop.lower()
                map_id = [map_ids[i] for i, x in enumerate(stop_descriptive_names) if x == stopl]
                if not map_id:
                    map_id = [map_ids[i] for i, x in enumerate(stop_names) if x == stopl]
                    if not map_id:
                        map_id = [map_ids[i] for i, x in enumerate(station_names) if x == stopl]
                        if not map_id:
                            print('Stop "' + stop + '" cannot be found.')
                            continue
                map_id = map_id[0]
                filt = self.stops == map_id
            else:
                filt = self.stops == stop
                map_id = stop

            if not filt.any():
                print('stop ' + str(stop) + ' appears to not exist.')
                continue

            station_name = self.station_name(map_id, 1)

            time_x = self.data[:,filt].squeeze()
            y = np.absolute(np.fft.rfft(time_x))
            y = y / np.sum(y)
            x = np.fft.rfftfreq(time_x.size, 1)

            legend_labels.append(station_name)
            line = plt.plot(x, y, axes=ax, gid=station_name)
            lines = lines + line

        plt.legend(legend_labels)

        return ax, lines

    def station_name(self, map_id, name_type=1):
        """
        Returns the station name of the given numerical ID

        Parameters
        ----------
        map_id :     the numerical ID as listed in the original CSV file.
        name_type  : type of name to be returned. 0 correponds to
                     the stop name, 1 corresponds to station name, and 2
                     corresponds to the descriptive station name.
                     DEFAULT: name_type = 1

        Returns
        -------
        name  : the station name
        """
        map_ids = map(lambda x: x[5], self.stop_data)

        try:
            i = map_ids.index(int(map_id))
        except ValueError:
            print('Cannot find a match for ' + str(map_id))
            return ''

        names = map(lambda x: x[2+name_type], self.stop_data)

        return names[i]

    def __generate_data_by_day(self, raw_data, raw_labels):
            stationNumbers = np.unique(raw_data[:, 0])
            dates = np.unique(raw_data[:, 1])
            dataByDay = np.zeros([len(dates), len(stationNumbers)])
            idx = 0
            L = raw_data.shape[0]
            for dateIdx, date in enumerate(dates):
                while idx < L and raw_data[idx, 1] == date:
                    stationNumberIdx = np.nonzero(stationNumbers == raw_data[idx, 0])[0]
                    dataByDay[dateIdx, stationNumberIdx] = raw_data[idx, 3]
                    idx = idx + 1
            return dataByDay, dates, stationNumbers

    def __read_train_data(self, filename='../data/L_station_daily_entry_totals.csv'):
        num_lines = sum(1 for line in open(filename))

        with open(filename) as f:
            line = f.readline().strip()
            labels = line.split(',')
            line = f.readline().strip().split(',')
            data = np.zeros([num_lines-1, 4])
            i = 0
            while line[0] != '':
                data[i, 0] = int(line[0])
                data[i, 1] = udt.ut(datetime.strptime(line[2], '%m/%d/%Y'))
                data[i, 2] = base36.tobase10(line[3])
                data[i, 3] = int(line[4])
                i = i + 1
                line = f.readline().strip().split(',')
            return data, labels

    def __get_L_stop_names(self, filename='../data/L_Stops.csv'):
        with open(filename) as f:
            labels = self.__parse_l_stops_list_line(f.readline().strip())
            line = self.__parse_l_stops_list_line(f.readline().strip())
            data = []
            while line[0] != '':
                stop_id                  = int(line[0])
                direction_id             = line[1]
                stop_name                = line[2]
                station_name             = line[3]
                station_descriptive_name = line[4]
                map_id                   = int(line[5])
                ada                      = self.__true_or_false(line[6])
                red                      = self.__true_or_false(line[7])
                blue                     = self.__true_or_false(line[8])
                g                        = self.__true_or_false(line[9])
                brn                      = self.__true_or_false(line[10])
                p                        = self.__true_or_false(line[11])
                pexp                     = self.__true_or_false(line[12])
                y                        = self.__true_or_false(line[13])
                pnk                      = self.__true_or_false(line[14])
                o                        = self.__true_or_false(line[15])
                location                 = line[16]
                data = data + [[stop_id, direction_id, stop_name, station_name,
                                station_descriptive_name, map_id, ada, red, blue,
                                g, brn, p, pexp, y, pnk, o, location]]
                line = self.__parse_l_stops_list_line(f.readline().strip())
            return data, labels

    def __true_or_false(self, txt):
        txtl = txt.lower()
        if txtl == 'true':
            return True
        elif txtl == 'false':
            return False
        else:
            raise(ValueError('Must be "true" or "false", given' + txtl))

    def __parse_l_stops_list_line(self, txt):
        return [x.strip('"') for x in re.split(r",+(?=[^()]*(?:\(|$))", txt)]
