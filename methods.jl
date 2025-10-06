using LinearAlgebra

# Helper function to shift an array with zero boundary conditions
function shift!(output, u, s)
    if s > 0
        copyto!(output, s+1, u, 1, length(u)-s)
        fill!(view(output, 1:s), zero(eltype(output)))
    elseif s < 0
        copyto!(output, 1, u, -s+1, length(u)+s)
        fill!(view(output, (length(u)+s+1):length(u)), zero(eltype(output)))
    else
        copyto!(output, u)
    end
    return output
end

function CrankNicolson1D!(u_out, u, μ, ∇μ, ∇2μ, D, dx, dt, kbT, temp_arrays)
    α = D / (4*dx*kbT)
    β = D / (2*dx^2)
    γ = D / (2*kbT)
    ϵ = D / (dx^2)

    # Use pre-allocated temporary arrays
    uP = temp_arrays.uP
    uN = temp_arrays.uN
    B = temp_arrays.B
    Adl = temp_arrays.Adl
    Adu = temp_arrays.Adu
    Au = temp_arrays.Au

    shift!(uP, u, 1)
    shift!(uN, u, -1)

    @. B = -1 * (uP*(-∇μ*α + β) + u*(∇2μ*γ - ϵ + 1/dt) + uN*(∇μ*α + β))

    @inbounds for i = 2:length(∇μ)
        Adl[i-1] = -∇μ[i]*α + β
    end
    @inbounds for i = 1:(length(∇μ)-1)
        Adu[i] = ∇μ[i]*α + β
    end
    @. Au = ∇2μ*γ - ϵ - 1/dt

    # Boundary conditions
    Adu[1] = 2*β
    Adl[end] = 2*β
    B[1] = -1*(uN[1]*2*β + u[1]*(∇2μ[1]*γ - ϵ + 1/dt))
    B[end] = -1*(uP[end]*2*β + u[end]*(∇2μ[end]*γ - ϵ + 1/dt))

    temp_arrays.A.dl .= Adl
    temp_arrays.A.d .= Au
    temp_arrays.A.du .= Adu

    ldiv!(u_out, temp_arrays.A, B)
    return u_out
end

struct TempArrays1D{T}
    uP::Vector{T}
    uN::Vector{T}
    B::Vector{T}
    Adl::Vector{T}
    Adu::Vector{T}
    Au::Vector{T}
    A::Tridiagonal{T,Vector{T}}
end

function TempArrays1D(::Type{T}, n::Int) where {T}
    uP = Vector{T}(undef, n)
    uN = Vector{T}(undef, n)
    B = Vector{T}(undef, n)
    Adl = Vector{T}(undef, n-1)
    Adu = Vector{T}(undef, n-1)
    Au = Vector{T}(undef, n)
    A = Tridiagonal(Adl, Au, Adu)
    return TempArrays1D(uP, uN, B, Adl, Adu, Au, A)
end

# 2D temporary arrays structure
struct TempArrays2D{T}
    temp_1d_x::Vector{TempArrays1D{T}}  # One for each row
    temp_1d_y::Vector{TempArrays1D{T}}  # One for each column
    u_temp_x::Vector{T}  # Temporary vector for x operations
    u_temp_y::Vector{T}  # Temporary vector for y operations
    u_half::Matrix{T}    # For ADI intermediate result
end

function TempArrays2D(::Type{T}, nx::Int, ny::Int) where {T}
    temp_1d_x = [TempArrays1D(T, ny) for _ = 1:nx]
    temp_1d_y = [TempArrays1D(T, nx) for _ = 1:ny]
    u_temp_x = Vector{T}(undef, ny)
    u_temp_y = Vector{T}(undef, nx)
    u_half = Matrix{T}(undef, nx, ny)
    return TempArrays2D(temp_1d_x, temp_1d_y, u_temp_x, u_temp_y, u_half)
end

function evolve_x!(u_new, u_2d, μ, ∇μX, ∇2μX, D, dx, dt, kbT, temp_arrays)
    nx = size(u_2d, 1)
    @inbounds for i = 1:nx
        copyto!(temp_arrays.u_temp_x, view(u_2d, i, :))

        CrankNicolson1D!(
            view(u_new, i, :),
            temp_arrays.u_temp_x,
            view(μ, i, :),
            view(∇μX, i, :),
            view(∇2μX, i, :),
            D,
            dx,
            dt,
            kbT,
            temp_arrays.temp_1d_x[i],
        )
    end
    return u_new
end

function evolve_y!(u_new, u_2d, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
    ny = size(u_2d, 2)
    @inbounds for j = 1:ny
        copyto!(temp_arrays.u_temp_y, view(u_2d, :, j))

        CrankNicolson1D!(
            view(u_new, :, j),
            temp_arrays.u_temp_y,
            view(μ, :, j),
            view(∇μY, :, j),
            view(∇2μY, :, j),
            D,
            dx,
            dt,
            kbT,
            temp_arrays.temp_1d_y[j],
        )
    end
    return u_new
end

function lie_splitting!(
    u,
    dt::Float64,
    dx,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    u_temp = temp_arrays.u_half  # Reuse the allocated matrix

    for step = 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
    end
    return u
end

function strang_splitting!(
    u,
    dt::Float64,
    dx,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    u_temp = temp_arrays.u_half

    for step = 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        copyto!(u, u_temp)
    end
    return u
end

struct ADITempArrays{T}
    u_half::Matrix{T}
    Adl::Vector{T}
    Adu::Vector{T}
    Au::Vector{T}
    B::Vector{T}
end

function ADITempArrays(::Type{T}, nx::Int, ny::Int) where {T}
    max_dim = max(nx, ny)
    u_half = Matrix{T}(undef, nx, ny)
    Adl = Vector{T}(undef, max_dim-1)
    Adu = Vector{T}(undef, max_dim-1)
    Au = Vector{T}(undef, max_dim)
    B = Vector{T}(undef, max_dim)
    return ADITempArrays(u_half, Adl, Adu, Au, B)
end

function x_direction_sweep!(
    u_half,
    u,
    αx,
    αy,
    βx,
    βy,
    dt,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    nx, ny = size(u)

    # Reuse pre-allocated arrays
    Adl = temp_arrays.Adl
    Adu = temp_arrays.Adu
    Au = temp_arrays.Au
    B = temp_arrays.B

    @inbounds for i = 1:nx
        # Resize views to ny elements
        Adl_view = view(Adl, 1:(ny-1))
        Adu_view = view(Adu, 1:(ny-1))
        Au_view = view(Au, 1:ny)
        B_view = view(B, 1:ny)

        fill!(Adl_view, 0)
        fill!(Adu_view, 0)
        fill!(Au_view, 0)
        fill!(B_view, 0)

        for j = 1:ny
            if j == 1
                # Left boundary
                if i == 1 || i == nx
                    if i == 1
                        y_term =
                            2*αy*(u[i+1, j] - u[i, j]) + u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                    else
                        y_term =
                            2*αy*(u[i-1, j] - u[i, j]) + u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                    end
                else
                    y_term =
                        αy*(u[i-1, j] - 2*u[i, j] + u[i+1, j]) +
                        βy*(u[i+1, j] - u[i-1, j])*∇μY[i, j] +
                        u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                end
                Au_view[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i, j]
                if j < ny
                    Adu_view[j] = -(2*αx + βx*∇μX[i, j])
                end
                B_view[j] = u[i, j] + y_term
            elseif j == ny
                # Right boundary
                if i == 1 || i == nx
                    if i == 1
                        y_term =
                            2*αy*(u[i+1, j] - u[i, j]) + u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                    else
                        y_term =
                            2*αy*(u[i-1, j] - u[i, j]) + u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                    end
                else
                    y_term =
                        αy*(u[i-1, j] - 2*u[i, j] + u[i+1, j]) +
                        βy*(u[i+1, j] - u[i-1, j])*∇μY[i, j] +
                        u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                end
                Adl_view[j-1] = -(2*αx - βx*∇μX[i, j])
                Au_view[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i, j]
                B_view[j] = u[i, j] + y_term
            else
                # Interior points
                if i == 1 || i == nx
                    if i == 1
                        y_term =
                            2*αy*(u[i+1, j] - u[i, j]) + u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                    else
                        y_term =
                            2*αy*(u[i-1, j] - u[i, j]) + u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                    end
                else
                    y_term =
                        αy*(u[i-1, j] - 2*u[i, j] + u[i+1, j]) +
                        βy*(u[i+1, j] - u[i-1, j])*∇μY[i, j] +
                        u[i, j]*D*dt/(2*kbT)*∇2μY[i, j]
                end
                Au_view[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i, j]
                Adl_view[j-1] = -αx - βx*∇μX[i, j]
                Adu_view[j] = -αx + βx*∇μX[i, j]
                B_view[j] = u[i, j] + y_term
            end
        end

        ldiv!(view(u_half, i, :), Tridiagonal(Adl_view, Au_view, Adu_view), B_view)
    end
    return u_half
end

function y_direction_sweep!(
    u,
    u_half,
    αx,
    αy,
    βx,
    βy,
    dt,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    nx, ny = size(u_half)

    # Reuse pre-allocated arrays
    Adl = temp_arrays.Adl
    Adu = temp_arrays.Adu
    Au = temp_arrays.Au
    B = temp_arrays.B

    @inbounds for j = 1:ny
        # Resize views to nx elements
        Adl_view = view(Adl, 1:(nx-1))
        Adu_view = view(Adu, 1:(nx-1))
        Au_view = view(Au, 1:nx)
        B_view = view(B, 1:nx)

        fill!(Adl_view, 0)
        fill!(Adu_view, 0)
        fill!(Au_view, 0)
        fill!(B_view, 0)

        for i = 1:nx
            if i == 1
                # Bottom boundary
                if j == 1 || j == ny
                    if j == 1
                        x_term =
                            2*αx*(u_half[i, j+1] - u_half[i, j]) +
                            u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                    else
                        x_term =
                            2*αx*(u_half[i, j-1] - u_half[i, j]) +
                            u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                    end
                else
                    x_term =
                        αx*(u_half[i, j-1] - 2*u_half[i, j] + u_half[i, j+1]) +
                        βx*(u_half[i, j+1] - u_half[i, j-1])*∇μX[i, j] +
                        u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                end
                Au_view[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i, j]
                if i < nx
                    Adu_view[i] = -(2*αy + βy*∇μY[i, j])
                end
                B_view[i] = u_half[i, j] + x_term
            elseif i == nx
                # Top boundary
                if j == 1 || j == ny
                    if j == 1
                        x_term =
                            2*αx*(u_half[i, j+1] - u_half[i, j]) +
                            u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                    else
                        x_term =
                            2*αx*(u_half[i, j-1] - u_half[i, j]) +
                            u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                    end
                else
                    x_term =
                        αx*(u_half[i, j-1] - 2*u_half[i, j] + u_half[i, j+1]) +
                        βx*(u_half[i, j+1] - u_half[i, j-1])*∇μX[i, j] +
                        u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                end
                Adl_view[i-1] = -(2*αy - βy*∇μY[i, j])
                Au_view[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i, j]
                B_view[i] = u_half[i, j] + x_term
            else
                # Interior points
                if j == 1 || j == ny
                    if j == 1
                        x_term =
                            2*αx*(u_half[i, j+1] - u_half[i, j]) +
                            u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                    else
                        x_term =
                            2*αx*(u_half[i, j-1] - u_half[i, j]) +
                            u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                    end
                else
                    x_term =
                        αx*(u_half[i, j-1] - 2*u_half[i, j] + u_half[i, j+1]) +
                        βx*(u_half[i, j+1] - u_half[i, j-1])*∇μX[i, j] +
                        u_half[i, j]*D*dt/(2*kbT)*∇2μX[i, j]
                end
                Au_view[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i, j]
                Adl_view[i-1] = -αy - βy*∇μY[i, j]
                Adu_view[i] = -αy + βy*∇μY[i, j]
                B_view[i] = u_half[i, j] + x_term
            end
        end

        ldiv!(view(u, :, j), Tridiagonal(Adl_view, Au_view, Adu_view), B_view)
    end
    return u
end

function adi_scheme!(
    u,
    dt::Float64,
    dx::Float64,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)

    # Main time-stepping loop
    for step = 1:num_steps
        # First sweep: x-direction (implicit in x, explicit in y)
        x_direction_sweep!(
            temp_arrays.u_half,
            u,
            αx,
            αy,
            βx,
            βy,
            dt,
            ∇μX,
            ∇μY,
            ∇2μX,
            ∇2μY,
            D,
            kbT,
            temp_arrays,
        )

        # Second sweep: y-direction (implicit in y, explicit in x)
        y_direction_sweep!(
            u,
            temp_arrays.u_half,
            αx,
            αy,
            βx,
            βy,
            dt,
            ∇μX,
            ∇μY,
            ∇2μX,
            ∇2μY,
            D,
            kbT,
            temp_arrays,
        )
    end

    return u
end

function fd!(u, dt::Float64, dx::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, u_new)
    nx, ny = size(u)
    alphax = D*dt/(dx^2)
    alphay = D*dt/(dx^2)
    betax = D*dt/(2*kbT*dx)
    betay = D*dt/(2*kbT*dx)

    for step = 1:num_steps
        @inbounds for i = 2:(nx-1), j = 2:(ny-1)
            u_new[i, j] =
                u[i, j] + (
                    alphax * (u[i-1, j] - 2 * u[i, j] + u[i+1, j]) +
                    alphay * (u[i, j-1] - 2 * u[i, j] + u[i, j+1]) +
                    betax * 2 * dx * (u[i, j] * ∇2μX[i, j]) +
                    betay * 2 * dx * (u[i, j] * ∇2μY[i, j]) +
                    betax * (u[i+1, j] - u[i-1, j]) * ∇μX[i, j] +
                    betay * (u[i, j+1] - u[i, j-1]) * ∇μY[i, j]
                )
        end

        # Boundary conditions
        @inbounds for i = 1:nx
            # Bottom and top boundaries
            u_new[i, 1] = u[i, 1] + alphay * (2 * u[i, 2] - 2 * u[i, 1])
            u_new[i, ny] = u[i, ny] + alphay * (2 * u[i, ny-1] - 2 * u[i, ny])
        end
        @inbounds for j = 1:ny
            # Left and right boundaries
            u_new[1, j] = u[1, j] + alphax * (2 * u[2, j] - 2 * u[1, j])
            u_new[nx, j] = u[nx, j] + alphax * (2 * u[nx-1, j] - 2 * u[nx, j])
        end

        u, u_new = u_new, u
    end
    return u
end

function lie_splitting(dt::Float64, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    nx, ny = size(u0)
    temp_arrays = TempArrays2D(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return lie_splitting!(
        u,
        dt,
        dx,
        num_steps,
        μ,
        ∇μX,
        ∇μY,
        ∇2μX,
        ∇2μY,
        D,
        kbT,
        temp_arrays,
    )
end

function strang_splitting(dt::Float64, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    nx, ny = size(u0)
    temp_arrays = TempArrays2D(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return strang_splitting!(
        u,
        dt,
        dx,
        num_steps,
        μ,
        ∇μX,
        ∇μY,
        ∇2μX,
        ∇2μY,
        D,
        kbT,
        temp_arrays,
    )
end

function adi_scheme(
    dt::Float64,
    dx::Float64,
    u0,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
)
    nx, ny = size(u0)
    temp_arrays = ADITempArrays(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return adi_scheme!(u, dt, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
end

function fd(dt::Float64, dx::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    u = copy(u0)  # Only copy needed for API compatibility
    u_new = similar(u)
    u_final = fd!(u, dt, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, u_new)
    # Ensure we return the correct array (due to swapping in fd!)
    return num_steps % 2 == 0 ? u_final : u_new
end
