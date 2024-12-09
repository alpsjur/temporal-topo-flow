using Oceananigans.Units

name = "brink_2010-prograde-switch_off-500m_res"

# forcing parameters 
switch = 10days
tmax   = 60days
dx     = 500meters 
dy     = 500meters


# define prograde forcing which switches off
function τy(x, y, t, u, v, h, p)
    if t < p.switch
        return -p.R*v/h - p.d*tanh(p.ω*t)/(p.ρ*h)
    else
        return -p.R*v/h
    end
end
