include("InputParameters.jl")
include("LBM-forces.jl")
include("Aster_PF.jl")
include("kernels.jl")


#   Intialize pop and boundaries
#   ==============================
f = CUDA.zeros(Tf, Nx, Nz, Q)
f_t = CUDA.zeros(Tf, Nx, Nz, Q)
ρ = CUDA.ones(Tf, Nx, Nz)*ρ0
v = CUDA.zeros(Tf, Nx, Nz, 2)

is_in = CUDA.ones(Bool, Nx, Nz)
Bpf = CUDA.zeros(Tf, Nx, Nz)
ΔBpf = CUDA.zeros(Tf, Nx, Nz)

ϕ = CUDA.zeros(Tf, Nx, Nz)
ϕ1 = CUDA.zeros(Tf, Nx, Nz)
ϕ2 = CUDA.zeros(Tf, Nx, Nz)
ϕt1 = CUDA.zeros(Tf, Nx, Nz)
ϕt2 = CUDA.zeros(Tf, Nx, Nz)
∇ϕ1 = CUDA.zeros(Tf, Nx, Nz, 2)
Δϕ1 = CUDA.zeros(Tf, Nx, Nz)
∇ϕ2 = CUDA.zeros(Tf, Nx, Nz, 2)
Δϕ2 = CUDA.zeros(Tf, Nx, Nz)

F = CUDA.zeros(Tf, Nx, Nz, 2)
σ = CUDA.zeros(Tf, Nx, Nz, 2, 2)
μ1 = CUDA.zeros(Tf, Nx, Nz)
μ2 = CUDA.zeros(Tf, Nx, Nz)

# CPU fields
v_cpu = zeros(Tf, Nx, Nz, 2)
ϕ1_cpu = zeros(Tf, Nx, Nz)
ϕ2_cpu = zeros(Tf, Nx, Nz)
Bpf_cpu = zeros(Tf, Nx, Nz)

dots_cpu = [cx_ch1 cz_ch1; cx_ch2 cz_ch2]

dots = CuArray{Float64}(dots_cpu)

initialize!(f, is_in, ϕ1, ϕ2, Bpf, ΔBpf, w, ρ0, 1.0e-2)

#   Time loop
#   ===========
CUDA.@time begin
    
    # Relaxation of phase-fileds (no growth)
    for tt = 1:1000
        evolve_asters!(ϕ1, ϕ2, ϕt1, ϕt2, σ, v, μ1, μ2, ∇ϕ1, Δϕ1, ∇ϕ2, Δϕ2, Bpf, 0.0)
    end
    
    for t=0:NΔt
        if t%prin==0
            println("Fmax = ", maximum(sqrt.(F[:,:,1].^2 .+ F[:,:,2].^2)))
            any(isnan, ρ) && return 1
            copyto!(v_cpu, v)
            copyto!(ϕ1_cpu, ϕ1)
            copyto!(ϕ2_cpu, ϕ2)
            copyto!(Bpf_cpu, Bpf)
            copyto!(dots_cpu, dots)
            save(string(dird,"data_",@sprintf("%09i",t),".jld"), "velocity", v_cpu, "pf1", ϕ1_cpu, "pf2", ϕ2_cpu, "Bpf", Bpf_cpu, "dots", Array(dots_cpu./Δx))
            CUDA.memory_status()
            println(" ")
            println(string(Int(round(t*Δt)),"s / ", duration, "s"))
        end
        
        ζΔμ = t*Δt > 0 ? ζΔμ0 : 0.0
        
        s1 = sum(ϕ1)
        s2 = sum(ϕ2)
        
        @views @. F[:,:,1] = (ϕ1/s1 - ϕ2/s2)*ζΔμ
        @views @. F[:,:,2] = 0.0
        
        
        evolve_asters!(ϕ1, ϕ2, ϕt1, ϕt2, σ, v, μ1, μ2, ∇ϕ1, Δϕ1, ∇ϕ2, Δϕ2, Bpf, kpoly)
        move_dots!(dots, v)
        
        comp_F!(F, σ)
        
        LBM_box!(f, f_t, ρ, v, F, is_in, w, ξ)
        # LBM_box_1st_order!(f, f_t, ρ, v, F, is_in, w, ξ)
        
    end
end