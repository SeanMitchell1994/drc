% A minimalistic sparse Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Matlab/Octave.
% from https://mantas.info/code/simple_esn/
% (c) 2012-2020 Mantas Lukosevicius
% Distributed under MIT license https://opensource.org/licenses/MIT

% load the data
trainLen = 784;
testLen = 784;
initLen = 25;
data = load('MackeyGlass_t17.txt');

% d2 = load('henon_y');
% data = d2.y_i';
% d2 = load('henon_x');
% data = d2.x_i';

data = load('mnist_5.mat');
data = data.d2';

% data2 = load('mnist_7.mat');
% data2 = data2.d2';

% d2 = load('chua.mat');
% data = d2.y(:,3);

% d2 = load('lorenz_x1');
% data = d2.x1;

% New shit
k = [0:1599];                % length of time series
y_i = zeros(1,length(k));   % Pre-allocation
y_i(1) = 0.7920;  % initial condition

% Solving the iterated solution
for i = 2:length(y_i)
    y_i(i) = (3.900142000000020 * y_i(i-1)) * (1 - y_i(i-1));
end
% y_i = y_i';
 d2 = y_i;
% d4 = load('Python/data/logistic_map_shaped.txt');
% plot some of it
figure(10);
plot(data);
title('A sample of data');

% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 40;
a = 0.3; % leaking rate
rand( 'seed', 42 );
Win = (rand(resSize,1+inSize)-0.5) .* 1;


% dense W:
W = rand(resSize,resSize)-0.5;
d3 = reshape(d2,resSize,[]);
W = W.*d3';

% sparse W:
% W = sprand(resSize,resSize,0.01);
% W_mask = (W~=0); 
% W(W_mask) = (W(W_mask)-0.5);

% normalizing and setting spectral radius
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* (1.25 / rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,length(data));
% set the corresponding target matrix directly
Yt = data';

% run the reservoir with the data and collect X
x1 = zeros(resSize,1);

x = zeros(resSize,1);

 z = load('E:\School\Graduate\2021 Spring\EE 510\Project\Datasets\mnist_train.csv');
for i = 1:60000
 
     data_train = z(i,(2:end));
     data_train = data_train';
    
    for t = 1:trainLen
        u = data_train(t);
        x1 = (1-a)*x1 + a*tanh( Win*[1;u] + W*x1 );
        x = x1;
        if t > initLen
            X(:,t-initLen) = [1;u;x];
        end
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
% data = load('mnist_7.mat');
% data = data.d2';
u = data(1);
for t = 1:testLen-1 
	x1 = (1-a)*x1 + a*tanh( Win*[1;u] + W*x1 );
    x = x1;
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
	u = y;
	% this would be a predictive mode:
	u = data(t+1);
end

% compute MSE for the first errorLen time steps
errorLen = 784;
mse = sum((data'-Y(1,:)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );
% 
% pe = sum(data'-Y(1,:))/data';

% plot some signals
figure(1);
plot( data, 'color', [0,0.75,0] );
hold on;
plot( Y', 'b' );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');

% figure(2);
% plot( X(1:20,1:200)' );
% title('Some reservoir activations x(n)');

figure(3);
bar( Wout' )
title('Output weights W^{out}');

figure(5);
output = reshape(Y',28,28);
input_check = reshape(data, 28, 28);
N = normalize(output);
imshowpair(input_check', output', 'montage');
%imshowpair(output', N', 'montage');
