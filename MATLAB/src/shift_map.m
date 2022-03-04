k = [0:0.01:1];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i2 = zeros(1,length(k));
ic = 0.4;
y_i(1) = ic;  % initial condition
y_i2(1) = ic;
a = 1;

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = a*mod(2*y_i(i - 1),1);
end

% % Solving the iterated solution
% for i = 2:length(y_i2)
%     if (0 < y_i2(i - 1) && y_i2(i - 1) < 0.5)
%         %a2x_n
%         y_i2(i) = y_i2(i - 1);
%     elseif (0.5 <= y_i2(i - 1) && y_i2(i - 1) < 1)
%         %a2x_n - 1
%         y_i2(i) = y_i2(i - 1)-1;
%     end
% end

clf;
%plot(y_i)
stem(y_i)
% hold on;
% stem(y_i2)