function [dates, rides] = routeRideTimelines(routeNums, data, toSmooth)
% [dates, rides] = routeRideTimelines(routeNums, data, toSmooth)
%
% Plots the number of rides per day for each route specified in routeNums
%
% INPUTS:
%   routeNums: string or cell array of strings specifying the routes
%   data     : Nx4 dataset. Optional, loaded if not given.
%              DEFAULT: data = load('cta_bus_rides_per_day.mat'); 
%   toSmooth : True to smooth dataset, false otherwise.
%              DEFAULT: toSmooth = false
%
% OUTPUTS:
%   dates: cell array of dates, one element for each route number specified
%   rides: cell array of number of people who rode the route, one element
%          for each route number specified.
%
% EXAMPLE:
%
% routeRideTimelines('2', [], true);
% routeRideTimelines({'171', '172'});
%

% Kevin Rose
% september, 2015

%% input handling
if nargin < 3 || isempty(toSmooth)
    toSmooth = false;
end
if nargin < 2 || isempty(data)
    data = load('cta_bus_rides_per_day.mat');
    data = data.data;
end
if nargin < 1 || isempty(routeNums)
    routeNums = dec2base(unique(data(:,1)), 36);
end

if ~iscell(routeNums) && ~ischar(routeNums) || ...
        (iscell(routeNums) && ~all(cellfun(@(x) ischar(x), routeNums)))
    error('routeNums must be a cell of strings or a string.')
end

if iscell(routeNums)
    routeNumLabels = routeNums;
    routeNums = cellfun(@(x) base2dec(x, 36), routeNums);
else
    routeNumLabels = routeNums;
    routeNums = base2dec(routeNums, 36);
end

%% initialize variables
lw = 1; % line width

N = size(data,1);
M = numel(routeNums);

X = cell(M,1);
Y = cell(M,1);

%% filters
isSpecifiedRoute = true(N,M);
for i = 1:M
    isSpecifiedRoute(:,i) = data(:,1) == routeNums(i);
end
% isWeekday = data(:,3) <= 3;

%% plot
figure;
hold on;
for i = 1:M
    ii = isSpecifiedRoute(:,i);
    
    X{i} = data(ii, 2);
    Y{i} = data(ii, 4);
    
    
    if toSmooth
        if numel(X{i}) < 2
            fprintf([routeNumLabels(i) '\n']);
            plot(X{i}, Y{i}, 'linewidth', lw);
        else
            plot(X{i}, csaps(X{i},Y{i},.01,X{i}), 'linewidth', lw);
        end
    else
        plot(X{i}, Y{i}, 'linewidth', lw);
    end
end
hold off

datetick2;

legend(routeNumLabels);

%% set outputs

dates = X;
rides = Y;

end

