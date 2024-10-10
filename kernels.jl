function initialize!(f, is_in, ϕ1, ϕ2, Bpf, ΔBpf, w, ρ0, D)
    initialize!(f, w, ρ0)
    fill_is_in!(is_in, x1, x2, z1, z2)
    build_bubbles!(ϕ1, ϕ2)   
    boundary_PF!(Bpf, ΔBpf, is_in, D)
    return nothing
end

function kernel_comp_F!(F, σ, Nx, Nz)
    # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    imm = mod(i-2,1:Nx); ipp = mod(i+2,1:Nx)
    jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    jmm = mod(j-2,1:Nz); jpp = mod(j+2,1:Nz)
    
    @inbounds begin
    
        ∂xσxx = (-σ[ipp,j,1,1] + 8*σ[ip,j,1,1] - 8*σ[i_,j,1,1] + σ[imm,j,1,1])/12
        ∂xσyx = (-σ[ipp,j,2,1] + 8*σ[ip,j,2,1] - 8*σ[i_,j,2,1] + σ[imm,j,2,1])/12
        
        ∂yσxy = (-σ[i,jpp,1,2] + 8*σ[i,jp,1,2] - 8*σ[i,jm,1,2] + σ[i,jmm,1,2])/12
        ∂yσyy = (-σ[i,jpp,2,2] + 8*σ[i,jp,2,2] - 8*σ[i,jm,2,2] + σ[i,jmm,2,2])/12

        F[i,j,1] += ∂xσxx + ∂yσxy
        F[i,j,2] += ∂xσyx + ∂yσyy
        
    end
    return nothing
end

comp_F!(F, σ) = @cuda threads = a2D_block blocks = a2D_grid kernel_comp_F!(F, σ, Nx, Nz)

function kernel_Bpf!(Bpf, is_in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if is_in[i,j]
        Bpf[i,j] = 2.0
    end
    return nothing
end

function kernel_ΔBpf!(ΔBpf, Bpf, Nx, Nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    
    @inbounds begin
        ΔBpf[i,j] = Bpf[i_,j] + Bpf[i,jm] - 4*Bpf[i,j] + Bpf[ip,j] + Bpf[i,jp]
    end
    
    return nothing
end

function boundary_PF!(Bpf, ΔBpf, is_in, D)
    @cuda threads = a2D_block blocks = a2D_grid kernel_Bpf!(Bpf, is_in)
    for i=1:100
        @cuda threads = a2D_block blocks = a2D_grid kernel_ΔBpf!(ΔBpf, Bpf, Nx, Nz)
        @. Bpf += D*ΔBpf
    end
    @cuda threads = a2D_block blocks = a2D_grid kernel_ΔBpf!(ΔBpf, Bpf, Nx, Nz)
end

function kernel_move_dots!(dots, v)
    i = threadIdx().x
    
    @inbounds x = round(Int, dots[i,1])
    @inbounds y = round(Int, dots[i,2])
    
    @inbounds dots[i,1] += v[x,y,1]
    @inbounds dots[i,2] += v[x,y,2]
    
    return nothing
end
move_dots!(dots, v) =  @cuda threads = 2 kernel_move_dots!(dots, v)
