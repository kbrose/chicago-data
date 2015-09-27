f = fopen('CTA_-_Ridership_-_Bus_Routes_-_Daily_Totals_by_Route.csv');
labels = fgetl(f);
fgetl(f);

data = zeros(651600,4);
i = 1;
line = fgetl(f);
while ischar(line)
    commas = strfind(line, ',');
    data(i,1) = base2dec(line(1:commas(1)-1), 36);
    data(i,2) = datenum(line(commas(1)+1:commas(2)-1));
    dayType = line(commas(2)+1);
    if strcmp(dayType, 'A')
        data(i,3) = 1;
    elseif strcmp(dayType, 'U')
        data(i,3) = 2;
    else
        data(i,3) = 3;
    end
    data(i,4) = str2double(line(commas(3)+1:end));
    
    line = fgetl(f);
    i = i + 1;
end
