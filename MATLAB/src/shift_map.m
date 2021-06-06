k = [0:50];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = -3.1;  % initial condition

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = mod(2*y_i(i - 1),1);
end

plot(y_i)