syms x
r = 3.25;
p = r*x - r*x^2;
p1 = compose(p,p);
p1 = p1 - x;
roots = solve( p1 );

for k = 1:length(roots)
    Df = real(diff(p1));
    x = real(roots(k));
    Df = subs(Df);
    real(Df);

    if (Df > 1)
        fprintf("unstable\n")
    elseif (Df < 1)
        fprintf("stable\n")
    end
end