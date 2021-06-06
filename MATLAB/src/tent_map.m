k = [0:0.5:35];                 % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = 0.61;                 % initial condition
mu = 1.2;

% Solving the iterated solution
for i = 2:length(y_i)
    if (y_i(i - 1) < 0.5)
        y_i(i) = mu * y_i(i - 1);
    elseif (0.5 <= y_i(i - 1))
        y_i(i) = mu * (1 - y_i(i - 1));
    end
end

%plot(k,y_i)
stem(y_i)
