f = fopen('CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv');
labels = strsplit(fgetl(f),',');

data = zeros(747838,4);
i = 1;
line = fgetl(f);
while ischar(line)
    commas = strfind(line, ',');
    data(i,1) = str2double(line(1:commas(1)-1));
%     data(i,2) = base2dec(line(commas(1)+1:commas(2)-1), 36);
    data(i,2) = datenum(line(commas(2)+1:commas(3)-1));
    dayType = line(commas(3)+1);
    if strcmp(dayType, 'A')
        data(i,3) = 1;
    elseif strcmp(dayType, 'U')
        data(i,3) = 2;
    else
        data(i,3) = 3;
    end
    data(i,4) = str2double(line(commas(4)+1:end));
    
    line = fgetl(f);
    i = i + 1;
end
