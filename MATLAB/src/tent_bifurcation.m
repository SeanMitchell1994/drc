r = 0.5;
k = [0:300];

while r < 2
    x = zeros(1,length(k));
    x(1) = rand(1);
    
    for i = 2:length(k)
        if (x(i - 1) < 0.5)
            x(i) = r * x(i - 1);
        elseif (0.5 <= x(i - 1))
            x(i) = r * (1 - x(i - 1));
        end
    end
    
    for i = 50:length(k)
       plot(r,x(i),'.k');
       hold on;
    end
    r = r + 0.01;
end

title('Orbit Diagram for the Tent Map');
xlabel('r')
ylabel('f_i[k]')