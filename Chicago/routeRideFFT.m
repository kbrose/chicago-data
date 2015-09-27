function [dates, rides, ffts] = routeRideFFT(routeNums, data)
% [dates, rides, ffts] = routeRideFFT(routeNums, data)
%
% Plots the FFT of rides per day for each route specified in routeNums
%
% INPUTS:
%   routeNums: string or cell array of strings specifying the routes
%   data     : Nx4 dataset. Optional, loaded if not given.
%              DEFAULT: data = load('cta_bus_rides_per_day.mat'); 
%
% OUTPUTS:
%   dates: cell array of dates, one element for each route number specified
%   rides: cell array of number of people who rode the route, one element
%          for each route number specified.
%   ffts : cell array of the (frequency, amplitude) pairs of the FFT, one
%          element for each route number specified
%
% EXAMPLE:
%
% routeRideFFT('2', data);
% routeRideFFT({'171', '172'});
%

% Kevin Rose
% september, 2015

%% INPUT HANDLING
if nargin < 2 || isempty(data)
    data = load('cta_bus_rides_per_day.mat');
    data = data.data;
end
if nargin < 1 || isempty(routeNums)
    routeNums = 2;
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

%% INITIALIZE VARIABLES

N = size(data,1);
M = numel(routeNums);

X = cell(M,1);
Y = cell(M,1);
ffts = cell(M,1);

%% FILTER(S)
isSpecifiedRoute = true(N,M);
for i = 1:M
    isSpecifiedRoute(:,i) = data(:,1) == routeNums(i);
end

%% PLOT
figure;
hold on;
for i = 1:M
    ii = isSpecifiedRoute(:,i);
    
    X{i} = data(ii, 2);
    Y{i} = data(ii, 4);
    
    L = length(X{i});
    L2 = round(L/2);
    
    fftAmps = fft(Y{i});
    P2 = abs(fftAmps / L);
    P1 = P2(1:L2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = ((0:L2)/L)';
    plot(f, P1);
    
    ffts{i} = [f, P1];
end
hold off

if M > 1
    legend(routeNumLabels);
end

%% SET OUTPUTS
dates = X;
rides = Y;

end

