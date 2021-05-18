function [mse] = desn_batch(in_vec)
    % A minimalistic sparse Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Matlab/Octave.
% from https://mantas.info/code/simple_esn/
% (c) 2012-2020 Mantas Lukosevicius
% Distributed under MIT license https://opensource.org/licenses/MIT

% load the data
trainLen = 2000;
testLen = 2000;
initLen = 100;
data = load('MackeyGlass_t17.txt');
% d2 = load('lorenz_x1');
% data = d2.x1;
% d2 = load('chua.mat');
% data = d2.y(:,3);

k = [0:1599];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = in_vec(2);  % initial condition

% Solving the iterated solution
% 16-cycle logistic map
for i = 2:length(y_i)
    y_i(i) = (3.900142000000020 * y_i(i-1)) * (1 - y_i(i-1));
end
% y_i = y_i';
d2 = y_i;

% d3 = load('Python/data/logistic_map_shaped.txt');
% plot some of it
% figure(10);
% plot(data(1:1000));
% title('A sample of data');

% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 40;
% a = leakage; % leaking rate
a = in_vec(1);
rand( 'seed', 42 );
Win = (rand(resSize,1+inSize)-0.5) .* 1;
% W2 = (rand(resSize,1+inSize)-0.5) .* 1;
% W3 = (rand(resSize,1+inSize)-0.5) .* 1;
% W4 = (rand(resSize,1+inSize)-0.5) .* 1;

% dense W:
W = rand(resSize,resSize)-0.5;
d3 = reshape(d2,resSize,[]);
W = W.*d3';

% sparse W:
% W = sprand(resSize,resSize,0.01);
% W_mask = (W~=0); 
% W(W_mask) = (W(W_mask)-0.5);

% normalizing and setting spectral radius
% disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
% disp 'done.'
W = W .* (1.25 / rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x1 = zeros(resSize,1);
% x2 = zeros(resSize,1);
% x3 = zeros(resSize,1);
% x4 = zeros(resSize,1);
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t);
	x1 = (1-a)*x1 + a*tanh( Win*[1;u] + W*x1 );
%     x2 = (1-a)*x1 + (1-a)*x2 + a*tanh( W2*[1;u] + W*x2 );
%     x3 = (1-a)*x2 + (1-a)*x3 + a*tanh( W3*[1;u] + W*x3 );
%     x4 = (1-a)*x3 + (1-a)*x4 + a*tanh( W4*[1;u] + W*x4 );
    x = x1;
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output by ridge regression
reg = 1e-8;  % regularization coefficient
% direct equations from texts:
%X_T = X'; 
%Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% using Matlab mldivide solver:
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize, testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x1 = (1-a)*x1 + a*tanh( Win*[1;u] + W*x1 );
%     x2 = (1-a)*x1 + (1-a)*x2 + a*tanh( W2*[1;u] + W*x2 );
%     x3 = (1-a)*x2 + (1-a)*x3 + a*tanh( W3*[1;u] + W*x3 );
%     x4 = (1-a)*x3 + (1-a)*x4 + a*tanh( W4*[1;u] + W*x4 );
    x = x1;
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
	u = y;
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
end

% compute MSE for the first errorLen time steps
errorLen = 500;
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
% disp( ['MSE = ', num2str( mse )] );

end

