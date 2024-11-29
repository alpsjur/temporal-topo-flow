using Oceananigans.Units

name = "brink_2010-prograde"

tmax = 64*3days

# define retrograde forcing which switches off
function τy(x, y, t, u, v, h, p)
    return -p.R*v/h - (p.d/π)*tanh(p.ω*t)/(p.ρ*h)
end
