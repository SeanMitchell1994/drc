%% Peak finder
clear;
k = [0:1599];                   % length of time series
j = [0:150];                    % length of listening
starting_index = 100;           % where does listening start?
y_i = zeros(1,length(k));       % Pre-allocation of times series
y_i(1) = 0.1;                     % initial condition
% a = 3.564407266095;             % y = ax(1-x)
a = 3.2;
da = 0.00001;                  % a increment per iteration
peak_cur = 0;                   % highest peak found
peak_index = starting_index;                 % where was the last highest peak?
peak_diff = 0;                  % distance between two peaks 
                                % (the orbit lvl)
index_cur = 0;
peak_diff_max = 0;
a_max = 0;

zero_cycle = 0;
first_appeared = 0;

output = []
output = zeros(5,5);
row_index = 1;

while a < 4
    % Solving the iterated solution
    for i = 2:length(y_i)
        y_i(i) = (a * y_i(i-1)) * (1 - y_i(i-1));
    end

    % Search for peaks and find the orbit level using them
    peak_cur = y_i(peak_index);
    for i2 = 0:length(j)-1
        index_cur = starting_index + i2;
        val_cur = y_i(starting_index + i2);
        
        if ((abs(peak_cur - val_cur)) < 0.001)
            peak_diff = (starting_index + i2) - peak_index;
            
            if (peak_diff == zero_cycle)
                % Skip if no orbits are found
                continue
            else
                % found orbit, save data
                fprintf('a: %f, peak: %f, cur: %f, orbit: %f, index_cur: %f \n',a, peak_cur, val_cur, peak_diff, index_cur);
                output(row_index, 1) = a;
                output(row_index, 2) = peak_cur;
                output(row_index, 3) = val_cur;
                output(row_index, 4) = peak_diff;
                output(row_index, 5) = index_cur;
                row_index = row_index + 1;
                break
            end
        end
    end
    a = a + da;
end

%% processing output to find the first occurance of each orbit
first_occurance = [];
len = length(output(:,1));
first_occurance = zeros(len,2);

% walk through output array
for k1 = 1:len
    if ismember(output(k1,4), first_occurance)
        continue
    else
        first_occurance(k1,1) = output(k1,1);
        first_occurance(k1,2) = output(k1,4);
    end
end

first_occurance( all(~first_occurance,2), : ) = [];
first_occurance( :, all(~first_occurance,1) ) = [];
