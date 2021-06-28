[t,y] = ode45(@chua,[0 250],[0.7 0 0]);
plot3(y(:,1),y(:,2),y(:,3))
grid