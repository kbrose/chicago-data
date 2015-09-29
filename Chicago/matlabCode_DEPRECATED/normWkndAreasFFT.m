function [routes, normAreas] = normWkndAreasFFT(data)
% [routes, normAreas] = normWkndAreasFFT(data)
%
% Compute the ratio of the amplitude of the DFT nearest to 1/7
% (representing a week frequency in Hz) to the total area under the DFT.
%
% INPUTS:
%   data : Nx4 matrix. If not specified it is loaded.
%          DEFAULT: data = load('cta_bus_rides_per_day.mat');
%
% OUTPUTS:
%   routes    : Route numbers in the base 10 representation of their base
%               36 values.
%   normAreas : Corresponding normalized 1/week amplitude, sorted in
%               descending order.
%
% EXAMPLE:
%
% [routes, normAreas] = normWkndAreasFFT;
%

% Kevin Rose
% September, 2015

%% INPUT HANDLING
if nargin < 2 || isempty(data)
    data = load('cta_bus_rides_per_day.mat');
    data = data.data;
end

%% FFT CALCULATION
routes = unique(data(:,1));

normAreas = zeros(numel(routes),1);
i = 1;
for route = routes'
    ii = data(:,1) == route;
    X = data(ii, 2);
    Y = data(ii, 4);
    
    L = length(X);
    L2 = round(L/2);
    
    if L == 1
        continue
    end
    
    fftAmps = fft(Y);
    P2 = abs(fftAmps / L);
    P1 = P2(1:L2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = ((0:L2)/L)';

    [~, weekIdx] = min(abs(f - 1/7));
    normAreas(i) = P2(weekIdx) / trapz(f, P1);
    
    i = i + 1;
    
end

%% OUTPUT
[normAreas, ii] = sort(normAreas, 1, 'descend');
routes = routes(ii);

end