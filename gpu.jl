using CUDA
using LinearAlgebra
using CUDA.CUSPARSE

const cusparse_handle = CUSPARSE.handle()

# Pre–allocated workspace
mutable struct GPUWorkSpace{T}
    uP  :: CuArray{T,2}
    uN  :: CuArray{T,2}
    rhs :: CuArray{T,2}

    # Flattened tridiagonal for cusparse (*one* contiguous slice per column)
    dl :: CuArray{T,1}
    d  :: CuArray{T,1}
    du :: CuArray{T,1}

    cusparse_buffer :: CuArray{UInt8}
    bufsize         :: Csize_t
end

# One-time allocation
function setup_gpu_workspace(T::Type, nx::Int, ny::Int)
    # temporary 2-D views (no copies)
    uP  = CUDA.zeros(T, nx, ny)
    uN  = CUDA.zeros(T, nx, ny)
    rhs = CUDA.zeros(T, nx, ny)

    # flattened tridiagonals
    m  = nx               # size of each system (x-direction)
    nrhs = ny             # number of such systems
    dl  = CUDA.zeros(T, m * nrhs)
    d   = CUDA.zeros(T, m * nrhs)
    du  = CUDA.zeros(T, m * nrhs)

    # query required CUSPARSE buffer
    sz = Ref{Csize_t}(0)
    CUSPARSE.cusparseDgtsv2StridedBatch_bufferSizeExt(
        cusparse_handle,
        Cint(m), dl, d, du, rhs,
        Cint(nrhs), Cint(m), sz)
    buf = CuArray{UInt8}(undef, sz[])

    GPUWorkSpace(uP, uN, rhs, dl, d, du, buf, sz[])
end

# Kernel that builds all diagonals and RHS
function _build_system!(uP, uN, u, rhs,
                        dl, d, du,
                        ∇μ_gpu, ∇2μ_gpu,
                        α, β, γ, ϵ, inv_dt,
                        (nx, ny))
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    1 <= i <= nx || return
    1 <= j <= ny || return

    ∇μ   = ∇μ_gpu[i,j]
    ∇2μ  = ∇2μ_gpu[i,j]

    # uP = u[i-1, j],  uN = u[i+1, j]
    u_i  =  u[i,j]
    u_p  = i > 1 ? u[i-1,j] : 0.0   # zero padded at 0-boundary
    u_n  = i < nx ? u[i+1,j] : 0.0
    uP[i,j] = u_p
    uN[i,j] = u_n

    # build tridiagonal and RHS
    dl_ij = -∇μ * α + β
    d_ij  =  ∇2μ * γ - ϵ - inv_dt
    du_ij =  ∇μ * α + β

    if i == 1
        du_ij = 2β
    elseif i == nx
        dl_ij = 2β
    end

    idx = (j-1) * nx + i   # linear index in flattened slice
    dl[idx] = dl_ij
    d[idx] = d_ij
    du[idx] = du_ij

    rhs[i,j] = -(u_p*(-∇μ*α + β) +
                 u_i*(∇2μ*γ - ϵ + inv_dt) +
                 u_n*( ∇μ*α + β))
    if i == 1
        rhs[i,j] = -(u_n*2β + u_i*(∇2μ*γ - ϵ + inv_dt))
    elseif i == nx
        rhs[i,j] = -(u_p*2β + u_i*(∇2μ*γ - ϵ + inv_dt))
    end
    return nothing
end

function evolve_x_gpu!(u, work, μ_gpu, ∇μx_gpu, ∇2μx_gpu,
                       D, dx, dt, kbT)
    T = eltype(u)
    nx, ny = size(u)
    α  = D/(4*dx*kbT)
    β  = D/(2*dx^2)
    γ  = D/(2*kbT)
    ϵ  = D/(dx^2)
    inv_dt = 1/dt

    threads = (32, 8)
    blocks  = (nx ÷ threads[1] + 1, ny ÷ threads[2] + 1)

    @cuda threads=threads blocks=blocks _build_system!(
            work.uP, work.uN, u, work.rhs,
            work.dl, work.d, work.du,
            ∇μx_gpu, ∇2μx_gpu, α, β, γ, ϵ, inv_dt, (nx, ny))

    rhs_flat = reshape(work.rhs, :)   # same memory, no copy
    CUSPARSE.cusparseDgtsv2StridedBatch(
        cusparse_handle,
        Cint(nx), work.dl, work.d, work.du, rhs_flat,
        Cint(ny),  Cint(nx),  work.cusparse_buffer)

    copyto!(u, reshape(rhs_flat, nx, ny)')  # transpose back to (nx × ny)
    return
end

function evolve_y_gpu!(u, work, μ_gpu, ∇μy_gpu, ∇2μy_gpu,
                       D, dx, dt, kbT)
    T = eltype(u)
    nx, ny = size(u)

    α  = D/(4*dx*kbT)
    β  = D/(2*dx^2)
    γ  = D/(2*kbT)
    ϵ  = D/(dx^2)
    inv_dt = 1/dt

    uT  = PermutedDimsArray(u, (2,1))         # cheap view
    rhsT= PermutedDimsArray(work.rhs, (2,1))

    threads = (32,8)
    blocks  = (ny ÷ threads[1] + 1, nx ÷ threads[2] + 1)

    @cuda threads=threads blocks=blocks _build_system!(
            work.uP, work.uN, uT, rhsT,
            work.dl, work.d, work.du,
            PermutedDimsArray(∇μy_gpu, (2,1)),
            PermutedDimsArray(∇2μy_gpu, (2,1)),
            α, β, γ, ϵ, inv_dt, (ny, nx))

    rhs_flat = reshape(work.rhs, :)
    CUSPARSE.cusparseDgtsv2StridedBatch(
        cusparse_handle,
        Cint(ny), work.dl, work.d, work.du, rhs_flat,
        Cint(nx),  Cint(ny),  work.cusparse_buffer)

    copyto!(u, reshape(rhs_flat, ny, nx))
    return
end

function lie_splitting_gpu(dt::Float64, dx, u0_cpu, num_steps,
                           μ_cpu, ∇μx_cpu, ∇μy_cpu,
                           ∇2μx_cpu, ∇2μy_cpu,
                           D, kbT;
                           return_intermediate=false)
    u_gpu = CuArray{Float64}(u0_cpu)
    μ_gpu = CuArray{Float64}(μ_cpu)
    ∇μx_gpu = CuArray{Float64}(∇μx_cpu)
    ∇μy_gpu = CuArray{Float64}(∇μy_cpu)
    ∇2μx_gpu = CuArray{Float64}(∇2μx_cpu)
    ∇2μy_gpu = CuArray{Float64}(∇2μy_cpu)

    work = setup_gpu_workspace(Float64, size(u_gpu)...)

    results = return_intermediate ? [copy(Array(u_gpu))] : missing
    for step=1:num_steps
        evolve_x_gpu!(u_gpu, work, μ_gpu, ∇μx_gpu, ∇2μx_gpu, D, dx, dt, kbT)
        evolve_y_gpu!(u_gpu, work, μ_gpu, ∇μy_gpu, ∇2μy_gpu, D, dx, dt, kbT)

        return_intermediate && push!(results, copy(Array(u_gpu)))
    end
    return_intermediate && return (Array(u_gpu), results)
    return Array(u_gpu)
end

function strang_splitting_gpu(dt::Float64, dx, u0_cpu, num_steps,
                              μ_cpu, ∇μx_cpu, ∇μy_cpu,
                              ∇2μx_cpu, ∇2μy_cpu,
                              D, kbT;
                              return_intermediate=false)
    u_gpu = CuArray{Float64}(u0_cpu)
    μ_gpu = CuArray{Float64}(μ_cpu)
    ∇μx_gpu = CuArray{Float64}(∇μx_cpu)
    ∇μy_gpu = CuArray{Float64}(∇μy_cpu)
    ∇2μx_gpu = CuArray{Float64}(∇2μx_cpu)
    ∇2μy_gpu = CuArray{Float64}(∇2μy_cpu)

    work = setup_gpu_workspace(Float64, size(u_gpu)...)

    results = return_intermediate ? [copy(Array(u_gpu))] : missing
    for step=1:num_steps
        evolve_x_gpu!(u_gpu, work, μ_gpu, ∇μx_gpu, ∇2μx_gpu, D, dx, dt/2, kbT)
        evolve_y_gpu!(u_gpu, work, μ_gpu, ∇μy_gpu, ∇2μy_gpu, D, dx, dt,     kbT)
        evolve_x_gpu!(u_gpu, work, μ_gpu, ∇μx_gpu, ∇2μx_gpu, D, dx, dt/2, kbT)

        return_intermediate && push!(results, copy(Array(u_gpu)))
    end
    return_intermediate && return (Array(u_gpu), results)
    return Array(u_gpu)
end
