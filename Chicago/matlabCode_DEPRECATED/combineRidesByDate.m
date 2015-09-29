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
i = 1;
j = 0;
oldDate = 0;
while i < size(data,1)
    currDate = data(i,2);
    if currDate ~= oldDate
        oldDate = currDate;
        j = j + 1;
    end
    rides(j, routeNums == data(i,1)) = data(i,4);
    i = i+1;
end

end