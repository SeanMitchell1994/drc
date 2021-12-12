% Programs: Lyapunov exponents of a Lorenz system.
% Reference.
% A. Wolf, J. B. Swift, H. L. Swinney, and J. A. Vastano,
% Determining Lyapunov Exponents from a Time Series, Physica D,
% Vol. 16, pp. 285-317, 1985.
% You must read the above paper to understand how the program works.
% Lyapunov exponents for the Lorenz system below are:
% L_1 = 0.9022, L_2 = 0.0003, L_3 = -14.5691 when tend = 10,000.
%sigma = 10.0;
%rho = 28;
%beta = 8/3;
%dx1/dt = sigma*(x2 - x1);
%dx2/dt = x1*(rho - x3) - x2;
%dx3/dt = x1*x2 - beta*x3;

function [Texp, Lexp] = Lorenz_Lyapunov(n, rhs_ext_fcn, fcn_integrator, tstart, stept, tend, ystart, ioutp)
clc;
close all;
n = 3;
rhs_ext_fcn = @lorenz_ext;
fcn_integrator = @ode45;
tstart = 0;
stept = 0.1;
tend = 1000;
ystart = [1 1 1];
ioutp = 10;
n1 = n;
n2 = n1*(n1 + 1);
% Number of steps.
nits = round((tend - tstart)/stept);
% Memory allocation.
y = zeros(n2, 1);
cum = zeros(n1, 1);
y0 = y;
gsc = cum;
znorm = cum;
% Initial values.
y(1:n) = ystart(:);
for i = 1:n1
  y((n1 + 1)*i) = 1.0;
end
t = tstart;
% Main loop.
for iterlya = 1:nits
% Solutuion of extended ODE system.
[T, Y] = feval(fcn_integrator, rhs_ext_fcn, [t t + stept], y);
t = t + stept;
y = Y(size(Y, 1),:);
for i = 1:n1
for j = 1:n1
  y0(n1*i + j) = y(n1*j + i);
end
end
% Construct new orthonormal basis by Gram-Schmidt.
znorm(1) = 0.0;
for j = 1:n1
    znorm(1) = znorm(1) + y0(n1*j+1)^2;
end
znorm(1) = sqrt(znorm(1));
for j = 1:n1
    y0(n1*j + 1) = y0(n1*j + 1)/znorm(1);
end
for j = 2:n1
for k = 1:(j - 1)
gsc(k) = 0.0;
for l = 1:n1
    gsc(k) = gsc(k) + y0(n1*l + j)*y0(n1*l + k);
end
end
for k = 1:n1
for l = 1:(j - 1)
y0(n1*k + j) = y0(n1*k + j) - gsc(l)*y0(n1*k + l);
end
end
znorm(j) = 0.0;
for k = 1:n1 
    znorm(j) = znorm(j) + y0(n1*k+j)^2;
end
znorm(j)=sqrt(znorm(j));
for k = 1:n1 
    y0(n1*k + j) = y0(n1*k + j)/znorm(j);
end
end
% Update running vector magnitudes.
for k = 1:n1 
    cum(k) = cum(k) + log(znorm(k));
end
% Normalize exponent.
for k = 1:n1
lp(k) = cum(k)/(t - tstart);
end
% Output modification.
if iterlya == 1
Lexp = lp; 
Texp = t;
else
Lexp = [Lexp; lp];
Texp = [Texp; t];
end
for i = 1:n1
for j = 1:n1
y(n1*j + i) = y0(n1*i + j);
end
end
end
% Show the Lyapunov exponent values on the graph.
str1 = num2str(Lexp(nits, 1));
str2 = num2str(Lexp(nits, 2));
str3 = num2str(Lexp(nits, 3));
pl = plot(Texp, Lexp,'LineWidth',2);
pl(1).Color = 'b';
pl(2).Color = 'g';
pl(3).Color = 'r';
legend('\lambda_1', '\lambda_2','\lambda_3')
legend('Location','northeast')
title('Dynamics of Lyapunov Exponents');
text(205, 1.5, '\lambda_1 =   ','Fontsize',20);
text(220, 1.68, str1,'Fontsize',20);
text(205, -1, '\lambda_2 =   ','Fontsize',20);
text(220, -0.73, str2,'Fontsize',20);
text(205, -13.8, '\lambda_3 =   ','Fontsize',20);
text(220, -13.5, str3,'Fontsize',20);
xlabel('Time');
ylabel('Lyapunov Exponents');
axis([-1,300, -16,6]);
set(gca,'FontSize',20)
% End of plot
function f = lorenz_ext(t, X)
% Values of parameters.
sigma = 10;
rho = 28;
beta = 8/3;
x = X(1);
y = X(2);
z = X(3);
Y = [X(4), X(7), X(10);
X(5), X(8), X(11);
X(6), X(9), X(12)];
f = zeros(9, 1);
%Lorenz equation.
f(1) = sigma*(y - x);
f(2) = -x*z + rho*x - y;
f(3) = x*y - beta*z;
%Linearized system.
Jac = [-sigma, sigma, 0;
        rho-z, -1, -x;
            y, x, -beta];
%Variational equation.
f(4:12) = Jac*Y;

