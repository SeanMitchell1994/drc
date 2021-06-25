k = [0:0.1:1];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = 0.5;  % initial condition
a = 1;

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = 2*a*mod(y_i(i - 1),1);
end

plot(y_i)