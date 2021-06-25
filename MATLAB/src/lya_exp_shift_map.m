lyap=zeros(1,1000);
j=0;
for(r=0:0.001:2)
    xn1=rand(1);
    lyp=0;
    j=j+1;
    for(i=1:10000)
        xn=xn1;
        xn1 = r*2*mod((xn),1);
       %wait for transient
       if(i>300)
           % calculate the sum of logaritm
           lyp=lyp+log(2*r);
       end
    end
    %calculate lyapun
    lyp=lyp/10000;
    lyap(j)=lyp;
end
r=0:0.001:2;
plot(r,lyap);
xlabel('r')
ylabel('Lambda')
grid on;
%line([0,0], ylim, 'Color', 'k','LineStyle','--'); % Draw line for Y axis.
line(xlim, [0,0], 'Color', 'k','LineStyle','--'); % Draw line for X axis.
%set(gca,'ydir','reverse')
