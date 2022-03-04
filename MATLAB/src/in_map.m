k = [0:0.01:1];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i2 = zeros(1,length(k));
ic = 0.4;
y_i(1) = ic;  % initial condition
y_i2(1) = ic;
a = 2.3;

res_size = 100;           % nxn size of the reservoir
in_size = 1; 
out_size = 1;

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = a*mod(2*y_i(i - 1),1);
end

load('lorenz.mat')

var = 2900;
r_t = Lexp(var:var);
%r_t = rand(res_size,res_size)-0.5;
rand( 'seed', 42 );
W_in = (rand(res_size,1+in_size)-0.5) .* 1;
subplot(1,2,1)
plot(W_in)
title('Win')

W_in  = W_in*[1;r_t];
subplot(1,2,2)
plot(W_in)
title('Win*U')
% subplot(1,3,2)
% plot(r_t)
% title('r(t)')