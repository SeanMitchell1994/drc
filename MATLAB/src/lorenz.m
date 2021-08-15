function lorenz = lorenz(t, y, sigma, beta, rho)
    lorenz = zeros(3,1);
    
    lorenz(1) = sigma * (y(2) - y(1));          %dx
    lorenz(2) = y(1) * (rho - y(3)) - y(2);     %dy
    lorenz(3) = y(1)*y(2) - beta*y(3);          %dz
end