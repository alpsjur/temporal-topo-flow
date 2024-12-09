using Oceananigans.Units

name = "brink_2010-retrograde-switch_off-100m_res"

# forcing parameters 
# forcing parameters 
switch = 20days
tmax   = 40days
dx     = 100meters 
dy     = 100meters

Δt = 1second


# define retrograde forcing which switches off
function τy(x, y, t, u, v, h, p)
    if t < p.switch
        return -p.R*v/h + p.d*tanh(p.ω*t)/(p.ρ*h)
    else
        return -p.R*v/h
    end
end
