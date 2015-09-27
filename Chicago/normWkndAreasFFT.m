function [routes, normAreas] = normWkndAreasFFT(data)

if nargin < 2 || isempty(data)
    data = load('cta_bus_rides_per_day.mat');
    data = data.data;
end

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
    
%     jj = f > (1/7 - .001) & f < (1/7 + .001);
%     if sum(jj) == 1
%         normAreas(i) = P1(jj) / trapz(f, P1);
%     elseif sum(jj) == 0
%         normAreas(i) = 0;
%     else
%         normAreas(i) = trapz(f(jj), P1(jj)) / trapz(f, P1);
%     end
    [~, weekIdx] = min(abs(f - 1/7));
    normAreas(i) = P2(weekIdx) / trapz(f, P1);
    
    i = i + 1;
    
end

[normAreas, ii] = sort(normAreas, 1, 'descend');
routes = routes(ii);

end