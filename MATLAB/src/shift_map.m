k = [0:0.01:1];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = 0.6;  % initial condition
a = 2;

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = a*mod(2*y_i(i - 1),1);
end

%plot(y_i)
stem(y_i)