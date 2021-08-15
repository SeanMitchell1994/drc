%% Number crunching
t_start = 0;                    % start time
t_end = 100;                     % end time
dt = 0.01;                      % time step
t_total = [t_start:dt:t_end];   % total time range

sigma = 10;
beta = 8/3;
rho = 28;

% Solving for x_0 = 1
[t_total, y1] = ode45(@(t_total, y1) lorenz(t_total, y1, sigma, beta, rho), t_total, [1 0 0]);
x1 = y1(:,1);
z1 = y1(:,3);
y1 = y1(:,2);

%% Plotting
plot(x1,y1);        % Time series
title('ODE45 Simulation of Lorenz System');
ylabel('x(t)');
xlabel('t');