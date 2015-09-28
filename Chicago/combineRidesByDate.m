function [dates, rides, routeNums] = combineRidesByDate(data)

%% INPUT HANDLING
if nargin < 1 || isempty(data)
    data = load('cta_bus_rides_per_day.mat');
    data = data.data;
end

%% INITIALIZE
dates = unique(data(:,2));
routeNums = unique(data(:,1));

rides = zeros(numel(dates), numel(routeNums));

%% CALCULATE
i = 1
oldDate = 0;
while i < size(data,1)
    currDate = 
end

end