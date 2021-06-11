syms x
r = 3.25;
p = r*x - r*x^2;
p = compose(p,p);
p = compose(p,p);
p = p - x;
roots = solve(p);

for k = 1:length(roots)
    Df = real(diff(p));
    x = real(roots(k));
    Df = abs(subs(Df));
    real(Df);

    if (Df > 1)
        fprintf("%f unstable\n", x)
    elseif (Df < 1)
        fprintf("%f stable\n", x)
    end
end