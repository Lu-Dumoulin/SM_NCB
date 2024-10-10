#   Initialize
#   ============
function kernel_initialize_pop!(f, w, ρ0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    for α in eachindex(w)
        f[i,j,α] = w[α]*ρ0
    end
    
    return nothing
end
@inline initialize!(f, w, ρ0) = @cuda threads = a2D_block blocks = a2D_grid kernel_initialize_pop!(f, w, ρ0)

function kernel_create_boundary!(is_in, x1, x2, z1, z2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i > x2 || i < x1 || j > z2 || j < z1 
        is_in[i,j] = false
    end
    
    return nothing
end
@inline fill_is_in!(is_in, x1, x2, z1, z2) = @cuda threads = a2D_block blocks = a2D_grid kernel_create_boundary!(is_in, x1, x2, z1, z2)


#   Collide and stream
#   ====================
function kernel_collide!(fn, f, F, ρ_, v, is_in, ω, w, ξ)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds begin      
        # Density and velocity
        ρ = ρ_[i,j]; vx = v[i,j,1]; vy = v[i,j,2]; v2= (vx*vx+vy*vy)
        # Source term factor
        ω_ = (1-0.5*ω)
        Fx=ω_*F[i,j,1]; Fy=ω_*F[i,j,2]
        if is_in[i,j]
            for α = 1:9
                fn[i,j,α] = (1.0-ω)*f[i,j,α] + ω*(ρ*w[α]*(1 + 3*(ξ[α,1]*vx + ξ[α,2]*vy) - 1.5*v2 + 4.5*(ξ[α,1]*vx + ξ[α,2]*vy)^2)) +
                            w[α]*( Fx*(3*(ξ[α,1]-vx)+9*((ξ[α,1]*vx + ξ[α,2]*vy))*ξ[α,1]) +
                                   Fy*(3*(ξ[α,2]-vy)+9*((ξ[α,1]*vx + ξ[α,2]*vy))*ξ[α,2]) )
            end
        else
            for α = 1:9
                fn[i,j,α] = 0.0 #ρ0*w[α]
            end
        end
    end
    return nothing
end

function kernel_collide_1st_order!(fn, f, F, ρ_, v, is_in, ω, w, ξ)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds begin      
        # Density and velocity
        ρ = ρ_[i,j]; vx = v[i,j,1]; vy = v[i,j,2]; #v2= (vx*vx+vy*vy)
        # Source term factor
        ω_ = (1-0.5*ω)
        Fx=ω_*F[i,j,1]; Fy=ω_*F[i,j,2]
        if is_in[i,j]
            for α = 1:9
                fn[i,j,α] = (1.0-ω)*f[i,j,α] + ω*(ρ*w[α]*(1 + 3*(ξ[α,1]*vx + ξ[α,2]*vy) )) +
                            w[α]*( Fx*(3*(ξ[α,1]-vx)+9*((ξ[α,1]*vx + ξ[α,2]*vy))*ξ[α,1]) +
                                   Fy*(3*(ξ[α,2]-vy)+9*((ξ[α,1]*vx + ξ[α,2]*vy))*ξ[α,2]) )
            end
        else
            for α = 1:9
                fn[i,j,α] = 0.0 #ρ0*w[α]
            end
        end
    end
    return nothing
end

function kernel_collide!(fn, f, F, ρ_, v, ω, w, ξ)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds begin      
        # Density and velocity
        ρ = ρ_[i,j]; vx = v[i,j,1]; vy = v[i,j,2]; v2= (vx*vx+vy*vy)
        # Source term factor
        ω_ = (1-0.5*ω)
        Fx=ω_*F[i,j,1]; Fy=ω_*F[i,j,2]
        for α = 1:9
            fn[i,j,α] = (1.0-ω)*f[i,j,α] + ω*(ρ*w[α]*(1 + 3*(ξ[α,1]*vx + ξ[α,2]*vy) - 1.5*v2 + 4.5*(ξ[α,1]*vx + ξ[α,2]*vy)^2)) +
                        w[α]*( Fx*(3*(ξ[α,1]-vx)+9*((ξ[α,1]*vx + ξ[α,2]*vy))*ξ[α,1]) +
                               Fy*(3*(ξ[α,2]-vy)+9*((ξ[α,1]*vx + ξ[α,2]*vy))*ξ[α,2]) )
        end
    end
    return nothing
end

function kernel_collide_TRT_ϕ!(fn, f, F, ρ_, v, ϕ_, is_in, Λ, ν1, ν2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds begin      
        if is_in[i,j]
            # Density and velocity
            ρ = ρ_[i,j]; vx = v[i,j,1]; vy = v[i,j,2]; v2 = (vx*vx+vy*vy)
            ϕ = ϕ_[i,j]*0.5
            ϕ = ϕ > 1.0 ? 1.0 : ϕ
            # Viscosity
            ν = ν1*(1.0-ϕ) + ν2*ϕ
            ωp = 1.0/(3.0*ν + 0.5)
            ωm = 1.0/(Λ * 1/(1/ωp-0.5) + 0.5)
            
            # Source term factors
            ωp_ = (1-0.5*ωp)
            ωm_ = (1-0.5*ωm)
            Fx=F[i,j,1]/3.0; Fy=F[i,j,2]/3.0 # * 3 / 9
            
            feq0 = ρ*4/9*(1 - 1.5*v2)
            feq1 = ρ/9*(1 + 3*vx - 1.5*v2 + 4.5*vx^2)
            feq2 = ρ/9*(1 + 3*vy - 1.5*v2 + 4.5*vy^2)
            feq3 = ρ/9*(1 - 3*vx - 1.5*v2 + 4.5*vx^2)
            feq4 = ρ/9*(1 - 3*vy - 1.5*v2 + 4.5*vy^2)
            feq5 = ρ/36*(1 + 3*(vx + vy) - 1.5*v2 + 4.5*(vx + vy)^2)
            feq6 = ρ/36*(1 + 3*(-vx + vy) - 1.5*v2 + 4.5*(-vx + vy)^2)
            feq7 = ρ/36*(1 - 3*(vx + vy) - 1.5*v2 + 4.5*(vx + vy)^2)
            feq8 = ρ/36*(1 + 3*(vx - vy) - 1.5*v2 + 4.5*(vx - vy)^2)
            
            f0 = f[i,j,1]; f1 = f[i,j,2]; f2 = f[i,j,3]; f3 = f[i,j,4]; 
            f4 = f[i,j,5]; f5 = f[i,j,6]; f6 = f[i,j,7]; f7 = f[i,j,8]; f8 = f[i,j,9]
            
            fn[i,j,1] = f0 - ωp*(f0-feq0)                                        + 4.0*( ωp_*( Fx * (-vx)                + Fy * (-vy)                )                )
            fn[i,j,2] = f1 - ωp*0.5*(f1+f3-feq1-feq3) - ωm*0.5*(f1-f3-feq1+feq3) +       ωp_*( Fx * (-vx + 3*vx)         + Fy * (-vy)                ) + ωm_*(Fx)
            fn[i,j,3] = f2 - ωp*0.5*(f2+f4-feq2-feq4) - ωm*0.5*(f2-f4-feq2+feq4) +       ωp_*( Fx * (-vx)                + Fy * (-vy  + 3*vy)        ) + ωm_*(Fy)
            fn[i,j,4] = f3 - ωp*0.5*(f1+f3-feq1-feq3) - ωm*0.5*(f3-f1-feq3+feq1) +       ωp_*( Fx * (-vx + 3*vx)         + Fy * (-vy)                ) + ωm_*(-Fx)
            fn[i,j,5] = f4 - ωp*0.5*(f2+f4-feq2-feq4) - ωm*0.5*(f4-f2-feq4+feq2) +       ωp_*( Fx * (-vx)                + Fy * (-vy + 3*vy)         ) + ωm_*(-Fy)
            fn[i,j,6] = f5 - ωp*0.5*(f5+f7-feq5-feq7) - ωm*0.5*(f5-f7-feq5+feq7) + 0.25*(ωp_*( Fx * (-vx + 3*(vx + vy))  + Fy * (-vy + 3*(vx + vy))  ) + ωm_*(Fx+Fy)  )
            fn[i,j,7] = f6 - ωp*0.5*(f6+f8-feq6-feq8) - ωm*0.5*(f6-f8-feq6+feq8) + 0.25*(ωp_*( Fx * (-vx - 3*(-vx + vy)) + Fy * (-vy + 3*(-vx + vy)) ) + ωm_*(-Fx+Fy) )
            fn[i,j,8] = f7 - ωp*0.5*(f7+f5-feq7-feq5) - ωm*0.5*(f7-f5-feq7+feq5) + 0.25*(ωp_*( Fx * (-vx + 3*(vx + vy))  + Fy * (-vy + 3*(vx + vy))  ) - ωm_*(Fx+Fy)  )
            fn[i,j,9] = f8 - ωp*0.5*(f6+f8-feq6-feq8) - ωm*0.5*(f8-f6-feq8+feq6) + 0.25*(ωp_*( Fx * (-vx + 3*(vx + -vy)) + Fy * (-vy - 3*(vx + -vy)) ) + ωm_*(Fx-Fy)  )
        else
            for α = 1:9
                fn[i,j,α] = 0.0 #ρ0*w[α]
            end
        end
    end
    return nothing
end

# Stream 3D:
function kernel_stream!(f, fpc, ξ, Nx, Nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    for α=1:9
        ii = mod(i + ξ[α,1], 1:Nx)
        jj = mod(j + ξ[α,2], 1:Nz)
        f[ii, jj, α] = fpc[i,j,α]
    end
    
    return nothing
end

function kernel_apply_BB!(f, fpc, x1, x2, z1, z2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i==x1
        f[i,j,2] = fpc[i,j,4]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,9] = fpc[i,j,7]
    elseif i==x2
        f[i,j,4] = fpc[i,j,2]
        f[i,j,8] = fpc[i,j,6]
        f[i,j,7] = fpc[i,j,9]
    end
    if j==z1
        f[i,j,3] = fpc[i,j,5]
        f[i,j,7] = fpc[i,j,9]
        f[i,j,6] = fpc[i,j,8]
    elseif j==z2
        f[i,j,5] = fpc[i,j,3]
        f[i,j,8] = fpc[i,j,6]
        f[i,j,9] = fpc[i,j,7]
    end
    return nothing
end
        

#   Meso to macro
#   ===============
function kernel_comp_ρ_v!(ρ, v, f, ξ, F, is_in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if !is_in[i,j]
        @inbounds ρ[i,j] = 0
        @inbounds v[i,j,1]=0.0; v[i,j,2]=0.0
        return nothing
    end
    
    @inbounds ρ[i,j] = 0
    @inbounds v[i,j,1]=0.5*F[i,j,1]; v[i,j,2]=0.5*F[i,j,2]
    
    @inbounds for α = 1:9
        ρ[i, j] +=  f[i, j, α] 
        v[i, j, 1] +=  f[i, j, α] * ξ[α,1]
        v[i, j, 2] +=  f[i, j, α] * ξ[α,2]
    end
    
    @inbounds v[i,j,1] *= 1/ρ[i,j]
    @inbounds v[i,j,2] *= 1/ρ[i,j]
    
    return nothing
end

@inline comp_ρ_v!(ρ, v, f, ξ, F, is_in) = CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_comp_ρ_v!(ρ, v, f, ξ, F, is_in)

function kernel_comp_ρ_v!(ρ, v, f, ξ, F)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    @inbounds ρ[i,j] = 0
    @inbounds v[i,j,1]=0.5*F[i,j,1]; v[i,j,2]=0.5*F[i,j,2]
    
    @inbounds for α = 1:9
        ρ[i, j] +=  f[i, j, α] 
        v[i, j, 1] +=  f[i, j, α] * ξ[α,1]
        v[i, j, 2] +=  f[i, j, α] * ξ[α,2]
    end
    
    @inbounds v[i,j,1] *= 1/ρ[i,j]
    @inbounds v[i,j,2] *= 1/ρ[i,j]
    
    return nothing
end

@inline comp_ρ_v!(ρ, v, f, ξ, F) = CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_comp_ρ_v!(ρ, v, f, ξ, F)

#   LBM function
#   ==============
function LBM_periodic!(f, f_t, ρ, v, F, w, ξ)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_collide!(f_t, f, F, ρ, v, ω, w, ξ)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_stream!(f, f_t, ξ, Nx, Nz)
    comp_ρ_v!(ρ, v, f, ξ, F)
end

function LBM_box!(f, fpc, ρ, v, F, is_in, w, ξ)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_collide!(fpc, f, F, ρ, v, is_in, ω, w, ξ)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_stream!(f, fpc, ξ, Nx, Nz)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_apply_BB!(f, fpc, x1, x2, z1, z2)
    comp_ρ_v!(ρ, v, f, ξ, F, is_in)
end
    
function LBM_box_1st_order!(f, fpc, ρ, v, F, is_in, w, ξ)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_collide_1st_order!(fpc, f, F, ρ, v, is_in, ω, w, ξ)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_stream!(f, fpc, ξ, Nx, Nz)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_apply_BB!(f, fpc, x1, x2, z1, z2)
    comp_ρ_v!(ρ, v, f, ξ, F, is_in)
end

function LBM_TRT_ϕ_box!(f, fpc, ρ, v, F, is_in, ϕ, Λ, ν1, ν2)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_collide_TRT_ϕ!(fpc, f, F, ρ, v, ϕ, is_in, Λ, ν1, ν2)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_stream!(f, fpc, ξ, Nx, Nz)
    CUDA.@sync @cuda threads = a2D_block blocks = a2D_grid kernel_apply_BB!(f, fpc, x1, x2, z1, z2)
    comp_ρ_v!(ρ, v, f, ξ, F, is_in)
end