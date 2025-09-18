using LinearAlgebra
using SparseArrays
using Plots

D = 0.01
nx = 100

"""
Solve 2D Smoluchowski equation using operator splitting with fully implicit scheme
Parameters:
- u0: Initial concentration field (2D array)
- μ: Potential field (2D array)
- D: Diffusion coefficient
- dx: Spatial step
- dt: Time step
- kbT: Boltzmann constant * Temperature
- num_steps: Number of time steps
"""

function shift(u, s)
    n = circshift(u, s)
    if s > 0
        n[1:s] .= 0
    end
    if s < 0
        n[end+s+1:end] .= 0
    end
    return n
end

function CrankNicolson(u, μ, ∇μ, ∇2μ, D, dx, dt, kbT)
    α = D / (4*dx*kbT)
    β = D / (2*dx^2)
    γ = D / (2*kbT)
    ϵ = D / (dx^2)
    uP = shift(u, 1)
    uN = shift(u, -1)
    B = -1 .* (uP.*(-∇μ.*α .+ β) .+ u.*(∇2μ*γ .- ϵ .+ 1/dt) .+ uN.*(∇μ.*α .+ β))
    Adl = -∇μ[2:end]*α .+ β
    Adu = ∇μ[1:end-1]*α .+ β
    Au = ∇2μ*γ .- ϵ .- 1/dt

    #Robni pogoji
    Adu[1] = 2*β
    Adl[end] = 2*β
    B[1] = -1*(uN[1]*2*β + u[1]*(∇2μ[1]*γ - ϵ + 1/dt))
    B[end] = -1*(uP[end]*2*β + u[end]*(∇2μ[end]*γ - ϵ + 1/dt))

    A = Tridiagonal(Adl, Au, Adu)
    
    return A \ B
end

dx = 6/100
μ = [sin(x*dx) + cos(y*dx) for y in 1:nx, x in 1:nx]
ΔμX = [2*pi*dx*cos(2*pi*x*dx) for y in 1:nx, x in 1:nx]
ΔΔμX = [-4*pi^2 * dx^2*sin(2*pi*x*dx) for y in 1:nx, x in 1:nx]
ΔμY = [-dx*sin(y*dx) for y in 1:nx, x in 1:nx]
ΔΔμY = [-dx^2*cos(y*dx) for y in 1:nx, x in 1:nx]

function test(d)
    R = (ones(nx))
    t = plot()
    for i in 1:d
        R = CrankNicolson(R, μ, ΔμX[1,:], ΔΔμX[1,:], D, dx, 0.01, 0.65)
        #println(R)
        #println(size(R))
    end
    return R
end
