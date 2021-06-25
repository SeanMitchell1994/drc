lyap=zeros(1,1000);
j=0;
for(r=3:0.001:4)
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
           lyp=lyp+log(abs(r-2*r*xn1));
       end
    end
    %calculate lyapun
    lyp=lyp/10000;
    lyap(j)=lyp;
end
r=3:0.001:4;
plot(r,lyap);
set(gca,'ydir','reverse')
