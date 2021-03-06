k = [0:1599];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = 0.7920;  % initial condition
cycles = 8;
x = [0:cycles];
y = [0:cycles];
r = 3.6870;
%r = 3.568640000010277;

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = r * y_i(i-1) - r * y_i(i-1).^2;
end

relative_index = 1;
for j = 300:(300 + cycles)
    
    x(relative_index) = y_i(j);
    relative_index = relative_index + 1;
end

for i = 1:length(x)
    y(i)= r - (2*r * x(i));
end

f = 1;
for i = 2:length(y)
    f = f * y(i);
end

f = abs(f)
% Less than 1 = stable