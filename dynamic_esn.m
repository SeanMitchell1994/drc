%% sys_init

% runtime parameters
% =============================================
clear;
train_len = 2000;       % length of training interval
test_len = 1000;        % length of testing interval
init_len = 100;         % warm up delay before training starts
a = 0.208;              % leaking rate/learning rate
in_size = 1; 
out_size = 1;
res_size = 40;           % nxn size of the reservoir

% runtime flags
% =============================================
run_generation = true;  % what output mode are we doing?
run_silent = false;
sparse_rev = false; 
dynamic_rev = true;

% Data load
% =============================================
% load reservoir function, r(t) from a file
% this should be a nxn matrix that matches the size of res_size
r_t = load('../../Datasets/logistic_map_shaped.txt');

% d2 = load('lorenz_x1');
% d3 = d2.x1(1:1600);
% r_t = reshape(d3,res_size,[]);


% load the training data
data = load('../../Datasets/MackeyGlass_t17.txt');

% d2 = load('henon_y');
% data = d2.y_i';
% d2 = load('henon_x');
% data = d2.x_i';

% d2 = load('chua.mat');
% data = d2.y(:,3);
% 
% d2 = load('lorenz_x1');
% data = d2.x1;
% d2 = load('lorenz_y1');
% data = d2.y1;
% d2 = load('lorenz_z1');
% data = d2.z1;

% reservoir prep
% =============================================
% generate W_in weight vector (mapping of input into reservoir)
rand( 'seed', 42 );
W_in = (rand(res_size,1+in_size)-0.5) .* 1;

% generate W weight vector (internal reservoir connections)
if (sparse_rev == false)
    % dense reservoir codepath
    % dense = many internal connections
    W = rand(res_size,res_size)-0.5;
    if (dynamic_rev == true)
        W = W.*r_t';
    end
else
    % sparse reservoir codepath
    % sparse = few internal connections
    W = sprand(res_size,res_size,0.01);
%     if (dynamic_rev == true)
%         W = W.*r_t';
%     end
    W_mask = (W~=0); 
    W(W_mask) = (W(W_mask)-0.5);
    if (dynamic_rev == true)
        W = W.*r_t';
    end
end

% normalizing and setting spectral radius
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* (1.25 / rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+in_size+res_size,train_len-init_len);

% set the corresponding target matrix directly
Yt = data(init_len+2:train_len+1)';

%% Training
% run the reservoir with the data and collect X
x = zeros(res_size,1);
for t = 1:train_len
	u = data(t);
	x = (1-a)*x + a*tanh( W_in*[1;u] + W*x );

	if t > init_len
		X(:,t-init_len) = [1;u;x];
	end
end

% train the output by ridge regression
reg = 1e-8;  % regularization coefficient
% direct equations from texts:
X_T = X'; 
Wout = Yt*X_T * inv(X*X_T + reg*eye(1+in_size+res_size));
% using Matlab mldivide solver:
%Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 


%% Testing
% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(out_size, test_len);
u = data(train_len+1);
for t = 1:test_len 
	x = (1-a)*x + a*tanh( W_in*[1;u] + W*x );

	y = Wout*[1;u;x];
	Y(:,t) = y;
    
    if (run_generation == true)
        % generative mode:
        u = y;
    else
        % this would be a predictive mode:
        u = data(train_len+t+1);
    end
end

% compute MSE for the first errorLen time steps
errorLen = 500;
mse = sum((data(train_len+2:train_len+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );


%% Plotting

if (run_silent == false)
    % plot some of it
    figure(10);
    plot(data(1:1000));
    title('A sample of data');

    % plot some signals
    figure(1);
    plot( data(train_len+2:train_len+test_len+1), 'color', [0,0.75,0] );
    hold on;
    plot( Y', 'b' );
    hold off;
    axis tight;
    title('Target and generated signals y(n) starting at n=0');
    legend('Target signal', 'Free-running predicted signal');

    figure(2);
    plot( X(1:end,1:end)' );
    title('Some reservoir activations x(n)');

    figure(3);
    bar( Wout' )
    title('Output weights W^{out}');
    
    figure(4);
    errorbar(data(train_len+2:train_len+errorLen+1)',Y(1,1:errorLen),'x');
    title('Error Bars');
end