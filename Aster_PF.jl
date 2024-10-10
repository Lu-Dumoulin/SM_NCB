# Initialize two phase-fields (ϕ1, ϕ2) of radius r_aster and center (c1x, c1y, c2x, c2y)
function kernel_asters_ini!(ϕ1, ϕ2, r_aster, c1x, c1y, c2x, c2y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    d1 = sqrt((i-c1x)^2 + (j-c1y)^2)
    d2 = sqrt((i-c2x)^2 + (j-c2y)^2)
    
    if d1<r_aster && d1<d2
        ϕ1[i,j] = 2.0
    elseif d2<r_aster && d2<d1
        ϕ2[i,j] = 2.0
    end
    
    return nothing
end

build_bubbles!(ϕ1, ϕ2) = @cuda threads = a2D_block blocks = a2D_grid kernel_asters_ini!(ϕ1, ϕ2, r_aster, cx_aster1, cz_aster1, cx_aster2, cz_aster2)


# Compute derivative
function kernel_comp_derivative!(∇ϕ1, Δϕ1, ∇ϕ2, Δϕ2, ϕ1, ϕ2, Nx, Nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    imm = mod(i-2,1:Nx); ipp = mod(i+2,1:Nx)
    jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    jmm = mod(j-2,1:Nz); jpp = mod(j+2,1:Nz)
    
    @inbounds begin
    
        Δϕ1[i,j] = ϕ1[i_,j] + ϕ1[ip,j] + ϕ1[i,jm] + ϕ1[i,jp] - 4*ϕ1[i,j]
        ∇ϕ1[i,j,1] = (-ϕ1[ipp,j] + 8*ϕ1[ip,j] - 8*ϕ1[i_,j] + ϕ1[imm,j])/12 #∂xϕ1
        ∇ϕ1[i,j,2] = (-ϕ1[i,jpp] + 8*ϕ1[i,jp] - 8*ϕ1[i,jm] + ϕ1[i,jmm])/12 #∂yϕ1
        
        Δϕ2[i,j] = ϕ2[i_,j] + ϕ2[ip,j] + ϕ2[i,jm] + ϕ2[i,jp] - 4*ϕ2[i,j]
        ∇ϕ2[i,j,1] = (-ϕ2[ipp,j] + 8*ϕ2[ip,j] - 8*ϕ2[i_,j] + ϕ2[imm,j])/12 #∂xϕ2
        ∇ϕ2[i,j,2] = (-ϕ2[i,jpp] + 8*ϕ2[i,jp] - 8*ϕ2[i,jm] + ϕ2[i,jmm])/12 #∂yϕ2
        
    end
    return nothing
end

# Compute stress
function kernel_comp_σ!(σ, μ1, μ2, ϕ1_, ∇ϕ1, Δϕ1_, ϕ2_, ∇ϕ2, Δϕ2_, a, kphi, λ, Nx,  Nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin
        
        ϕ1 = ϕ1_[i,j]; ϕ12 = ϕ1*ϕ1
        ∂xϕ1 = ∇ϕ1[i,j,1]; ∂yϕ1 = ∇ϕ1[i,j,2]
        Δϕ1 = Δϕ1_[i,j]
        
        ϕ2 = ϕ2_[i,j]; ϕ22 = ϕ2*ϕ2
        ∂xϕ2 = ∇ϕ2[i,j,1]; ∂yϕ2 = ∇ϕ2[i,j,2]
        Δϕ2 = Δϕ2_[i,j]

        μ1[i,j] = a*ϕ1*(ϕ1-2.0)*(ϕ1-1.0) - kphi*Δϕ1 + λ*ϕ1*ϕ22
        μ2[i,j] = a*ϕ2*(ϕ2-2.0)*(ϕ2-1.0) - kphi*Δϕ2 + λ*ϕ2*ϕ12

        # f - ϕμ
        f_ϕiμi = 0.25*a*ϕ12*(ϕ1-2.0)^2 + 0.5*kphi*(∂xϕ1*∂xϕ1+∂yϕ1*∂yϕ1) - ϕ1*μ1[i,j] +
                0.25*a*ϕ22*(ϕ2-2.0)^2 + 0.5*kphi*(∂xϕ2*∂xϕ2+∂yϕ2*∂yϕ2) - ϕ2*μ2[i,j] +
                λ*ϕ12*ϕ22*0.5

        σ[i,j,1,1] = f_ϕiμi - kphi*(∂xϕ1*∂xϕ1 + ∂xϕ2*∂xϕ2)
        σ[i,j,2,2] = f_ϕiμi - kphi*(∂yϕ1*∂yϕ1 + ∂yϕ2*∂yϕ2)
        σ[i,j,1,2] = -kphi*∂xϕ1*∂yϕ1 -kphi*∂xϕ2*∂yϕ2
        
        σ[i,j,2,1] = σ[i,j,1,2]  
    end
    return nothing
end

# Compute phase-field at t+1
function kernel_comp_ϕ!(ϕan, ϕa, ϕb, v, μa, ∇ϕa, Bpf, M, kpoly, Nx, Nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    
    @inbounds begin
        ∂xϕvx = -abs(v[i,j,1])*ϕa[i,j]
        ∂xϕvx += v[i_,j,1] > 0 ? ϕa[i_,j]*v[i_,j,1] : 0
        ∂xϕvx -= v[ip,j,1] < 0 ? ϕa[ip,j]*v[ip,j,1] : 0
        ∂yϕvy = -abs(v[i,j,2])*ϕa[i,j]
        ∂yϕvy += v[i,jm,2] > 0 ? ϕa[i,jm]*v[i,jm,2] : 0
        ∂yϕvy -= v[i,jp,2] < 0 ? ϕa[i,jp]*v[i,jp,2] : 0
        
        Δμa = μa[i_,j] + μa[i,jm] - 4*μa[i,j] + μa[ip,j] + μa[i,jp]
        ϕan[i,j] = ϕa[i,j] + ∂xϕvx + ∂yϕvy + M*Δμa + kpoly*sqrt((∇ϕa[i,j,1])^2+(∇ϕa[i,j,2])^2)*(Bpf[i,j]-1.9)*(1.0-ϕb[i,j])
    end
    return nothing
end

# Update phase-field t+1 -> t
function kernel_update_ϕ!(ϕ1, ϕ2, ϕt1, ϕt2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    @inbounds begin
        ϕ1[i,j] = ϕt1[i,j]
        ϕ2[i,j] = ϕt2[i,j]
    end
    return nothing
end


@inline function evolve_asters!(ϕ1, ϕ2, ϕt1, ϕt2, σ, v, μ1, μ2, ∇ϕ1, Δϕ1, ∇ϕ2, Δϕ2, Bpf, kpoly)
    @cuda threads = a2D_block blocks = a2D_grid kernel_comp_derivative!(∇ϕ1, Δϕ1, ∇ϕ2, Δϕ2, ϕ1, ϕ2, Nx, Nz)
    @cuda threads = a2D_block blocks = a2D_grid kernel_comp_σ!(σ, μ1, μ2, ϕ1, ∇ϕ1, Δϕ1, ϕ2, ∇ϕ2, Δϕ2, a, kphi, λ, Nx, Nz)
    @cuda threads = a2D_block blocks = a2D_grid kernel_comp_ϕ!(ϕt1, ϕ1, ϕ2, v, μ1, ∇ϕ1, Bpf, M, kpoly, Nx, Nz)
    @cuda threads = a2D_block blocks = a2D_grid kernel_comp_ϕ!(ϕt2, ϕ2, ϕ1, v, μ2, ∇ϕ2, Bpf, M, kpoly, Nx, Nz)
    @cuda threads = a2D_block blocks = a2D_grid kernel_update_ϕ!(ϕ1, ϕ2, ϕt1, ϕt2)
end
