%----------Chua.m----------
function chua = chua(t,in)

x = in(1);
y = in(2);
z = in(3);

alpha  = 15.6;
beta   = 28; 
m0     = -1.143;
m1     = -0.714;

h = m1*x+0.5*(m0-m1)*(abs(x+1)-abs(x-1));

xdot = alpha*(y-x-h);
ydot = x - y+ z;
zdot  = -beta*y;

chua = [xdot ydot zdot]';

end