%% Peak finder
clear;
k = [0:1599];                   % length of time series
j = [0:150];                    % length of listening
starting_index = 100;           % where does listening start?
y_i = zeros(1,length(k));       % Pre-allocation of times series
y_i(1) = 0.1;                     % initial condition

a = 3.3;
da = 0.000001;                  % a increment per iteration
peak_cur = 0;                   % highest peak found
peak_index = starting_index-1;                 % where was the last highest peak?
peak_diff = 0;                  % distance between two peaks 
                                % (the orbit lvl)
index_cur = 0;
peak_diff_max = 0;
a_max = 0;

array = zeros(5,2);

while a < 3.544090
    % Solving the iterated solution
    for i = 2:length(y_i)
        y_i(i) = (a * y_i(i-1)) * (1 - y_i(i-1));
    end

    % Search for peaks and find the orbit level using them
    peak_cur = y_i(peak_index);
    for i2 = 0:length(j)-1
        index_cur = starting_index + i2;
        val_cur = y_i(starting_index + i2);

        if ((abs(peak_cur - val_cur)) < 0.00001)
            peak_diff = (starting_index + i2) - peak_index;
            
%             if (peak_diff > peak_diff_max)
%                 peak_diff_max = peak_diff;
%                 a_max = a;
%             end
            fprintf('%f:%d peak found\n',a,peak_diff);
            array(1,1) = a;
            array(1,2) = peak_diff;
            
            if peak_diff == 4
                return
            end
            %return 
            break
        end

        if (val_cur > peak_cur)
            peak_cur = val_cur;
            peak_index = starting_index + i2;
        end
    end
    a = a + da;
    
end