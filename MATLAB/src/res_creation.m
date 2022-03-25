k = [0:0.01:1];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i2 = zeros(1,length(k));
ic = 0.4;
y_i(1) = ic;  % initial condition
y_i2(1) = ic;
a = 2.1;

res_size = 100;           % nxn size of the reservoir

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = a*mod(2*y_i(i - 1),1);
end

r_t = y_i(1:100);
%r_t = rand(res_size,res_size)-0.5;
rand( 'seed', 42 );
W = rand(res_size,res_size)-0.5;
% histogram(W);
% figure
subplot(2,2,1)
plot(W)
title('random seed [W]')
W = W.*r_t';

subplot(2,2,3)
plot(W)
title('W*r(t)')
subplot(2,2,2)
plot(r_t)
title('res fcn [r(t)]')


Fs = 100;           % Sampling frequency
t = -1:1/Fs:1;  % Time vector 
L = length(W);      % Signal length
Y = fft(W);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

subplot(2,2,4)
% plot(f,P1) 
% title('FFT of W*r(t)')
% xlabel('freq (Hz)')
% ylabel('Amplitude')
% plot(W_in + W)
% title('Win*U + Wr(t)')
% figure
histogram(W);

% figure;
% % plot(tanh(W_in + W))
% plot(std(W_in + W))