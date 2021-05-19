clear;
a = 0.001;
da = 0.001;
% a = 0.372;           % best performing leaking rate
y_0 = 0.001;            % okay so initial conditions don't matter
y_0 = 0.7920;
dy = 0.001;             % but what about number of orbits?
runs = 1000;
x = zeros(runs-2,3);
i = 0;

disp 'Running desn_bath...';
while a <= 0.999
    if y_0 >= 0.999
        break
    end
    
    mse = desn_batch([a,y_0]);
    x(i+1,1) = a;
    x(i+1,3) = y_0;
    x(i+1,2) = mse;
    a = a + da;
    %y_0 = y_0 + dy;
    i = i + 1;
end;
disp 'Done.';

plot(x(:,1),x(:,2));
title("MSE for a given leakage");
xlabel("Leaking rate");
ylabel("MSE");
[min_mse,index] = min(x(:,2));
ylim([-0.2 5]);

% index 540: a = 0.54, mse = 0.2194; y_i_a = 3.889;
% single layer, lorenz.x1 time series trained: a = 0.883;
% double layer, lorenz.x1 time series trained: a = 0.780;
