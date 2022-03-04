lyap=zeros(1,1000);
dt1 = zeros(1,1000);
j=0;
for(r=0:0.001:4)
    xn1=rand(1);
    lyp=0;
    j=j+1;
    for(i=1:10000)
        xn=xn1;
        %logistic map
        xn1=r*xn*(1-xn);
       %wait for transient
       if(i>300)
           % calculate th sum of logaritm
           dt = r-2*r*xn1;
           lyp=lyp+log(abs(r-2*r*xn1));
       end
    end
    %calculate lyapun
    lyp=lyp/10000;
    lyap(j)=lyp;
    dt1(j) = dt;
end
r=0:0.001:4;
plot(r,lyap);
grid on;
line(xlim, [0,0], 'Color', 'k','LineStyle','--'); % Draw line for X axis.
%set(gca,'ydir','reverse')

lyap2=zeros(1,1000);
dt2 = zeros(1,1000);
j=0;
for(r2=0:0.001:4)
    xn1=rand(1);
    lyp=0;
    j=j+1;
    for(i=1:10000)
        xn=xn1;
        xn1 = r2*2*mod((xn),1);
       %wait for transient
       if(i>300)
           % calculate the sum of logaritm
           dt = 2*r2;
           lyp=lyp+log(2*r2);
       end
    end
    %calculate lyapun
    lyp=lyp/10000;
    lyap2(j)=lyp;
    dt2(j) = dt;
end
r2=0:0.001:4;
figure
plot(r2,lyap2);
xlabel('r')
ylabel('Lambda')
grid on;
%line([0,0], ylim, 'Color', 'k','LineStyle','--'); % Draw line for Y axis.
line(xlim, [0,0], 'Color', 'k','LineStyle','--'); % Draw line for X axis.
%set(gca,'ydir','reverse')

figure
plot(dt2, dt1)
line(xlim, [0,0], 'Color', 'k','LineStyle','--'); % Draw line for X axis.
grid on;
