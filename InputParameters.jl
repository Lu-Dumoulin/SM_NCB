dird = joinpath(@__DIR__, "simulation_1/Data/")
mkpath(dird) 
using CUDA, JLD, Printf, FileIO

# type
Ti = Int64 # Int32
Tf = Float64 # Float32
Ta = CuArray;

D = 2
Q = 9

w = Ta([4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36])
# Lattice Velocities
ξ = Ta([0 0; 1 0; 0 1; -1 0; 0 -1; 1 1; -1 1; -1 -1; 1 -1]);

## Physical Units
Lz           = Tf(200)     # cell_lz/grid_lz
Lx           = Lz*1.4      # cell_lx/grid_lx
Δx           = 1.0         # μm
grid_lx      = Lx/Δx+20    # μm (approximate)
grid_lz      = Lz/Δx+20    # μm (approximate)
aster_growth = 0.35        # μm/s
chromo_velo  = 0.1         # μm/s
chromo_radi  = 5.0         # μm
aster_radi   = 20          # μm 
duration     = 300         # second
Re           = 1e-2        #
distance_chr = 18.0        #

## GPU constraints
a2D_block = (16, 16)
Bx::Int = floor(Int, grid_lx/16)
Bz::Int = floor(Int, grid_lz/16)

a2D_grid = (Bx, Bz)
Nx::Int = Bx * 16
Nz::Int = Bz * 16

@show grid_lx = Nx
@show grid_lz = Nz
@show cell_lx = Lx/Δx # μm
@show cell_lz = Lz/Δx # μm

## LBM Constraints
ρ0 = 1.0
vmax_LB = 1e-5 # δx/δt
Lc = 10*2/Δx       # δx
r_chromo = aster_radi/Δx
r_aster = aster_radi/Δx
V0 = 4/3*pi*r_aster^3*2.0
ε = 0.0
## Physical time-step
@show Δt = vmax_LB*Δx/chromo_velo # s

## LBM param
# FOR BGK
@show τ = 3*2*r_aster*vmax_LB/Re + 0.5 # 1/δt
# @show τ = 0.51 
ω=1/τ
@show visco = (2*τ-1)/6

# FOR TRT
@show const Λ = 3/16
const ν1 = visco
const ν2 = 1.0*visco

#   Duration, writing
#   –––––––––––––––––––
NΔt = Int(round(duration/Δt))  # δt
prin = Int(round(5/Δt))       # δt

#   Aster
#   ––––––––––––––
diff_adv_frac = Tf(1.0e6)
## ϕ
a = Tf(0.1)/diff_adv_frac
kphi = Tf(0.2)/diff_adv_frac
λ = Tf(0.01)/diff_adv_frac
M = Tf(0.10)*diff_adv_frac
D = M*a
kpoly = Tf(4)*Δt

## Active stress
ζΔμ0 = Tf(1.8e-5)

# Corners of boundary (as function of Lx, Ly, Lz) !!!!!!
x1 = floor(Int,(grid_lx-cell_lx)*0.5); x2 = floor(Int,grid_lx-x1)
z1 = floor(Int,(grid_lz-cell_lz)*0.5); z2 = floor(Int,grid_lz-z1)
@show x1, x2, z1, z2

cx = floor(Int, 0.5*(x2-x1)+x1)
cz = floor(Int, 0.5*(z2-z1)+z1)

daa = r_aster*0.5
global cx_aster1 = Int(floor(cx + daa))
global cz_aster1 = cz
global cx_aster2 = Int(floor(cx - daa))
global cz_aster2 = cz

dchch = distance_chr/Δx
global cx_ch1 = Int(floor(cx + dchch))
global cz_ch1 = cz
global cx_ch2 = Int(floor(cx - dchch))
global cz_ch2 = cz

Re_fact = Lc/visco

set_zero_subnormals(true)