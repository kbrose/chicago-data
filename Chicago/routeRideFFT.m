function [dates, rides, ffts] = routeRideFFT(routeNums, data)

if nargin < 2 || isempty(data)
    data = load('cta_bus_rides_per_day.mat');
    data = data.data;
end
if nargin < 1 || isempty(routeNums)
    routeNums = 2;
end

N = size(data,1);
M = numel(routeNums);

isSpecifiedRoute = true(N,M);
for i = 1:M
    isSpecifiedRoute(:,i) = data(:,1) == routeNums(i);
end

X = cell(M,1);
Y = cell(M,1);
ffts = cell(M,1);
figure;
hold on;
for i = 1:M
    ii = isSpecifiedRoute(:,i);
    
    X{i} = data(ii, 2);
    Y{i} = data(ii, 4);
    
    fftAmps = fft(Y{i});
    P2 = abs(fftAmps / length(X{i}));
    P1 = P2(1:length(X{i})/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = ((0:length(X{i})/2)/length(X{i}))';
    plot(f, P1);
    
    trapz(f, P1)
    
    ffts{i} = [f, P1];
end
hold off

if M > 1
    legend(strread(num2str(routeNums),'%s'));
end

dates = X;
rides = Y;

end

