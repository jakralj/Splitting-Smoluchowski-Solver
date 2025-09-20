function generate_potential_1(nx, dx)
    # Grid generation
    x_range = range(-1, 1, length=nx)
    y_range = range(-1, 1, length=nx)

    μ = zeros(nx, nx)
    ∇μX = zeros(nx, nx)
    ∇μY = zeros(nx, nx)
    ∇2μX = zeros(nx, nx)
    ∇2μY = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            # Potential: f(x,y) = x²y + e^x sin(y)
            μ[i,j] = x^2 * y + exp(x) * sin(y)

            # First derivatives
            ∇μX[i,j] = 2*x*y + exp(x)*sin(y)  # ∂f/∂x
            ∇μY[i,j] = x^2 + exp(x)*cos(y)    # ∂f/∂y

            # Second derivatives
            ∇2μX[i,j] = 2*y + exp(x)*sin(y)   # ∂²f/∂x²
            ∇2μY[i,j] = -exp(x)*sin(y)        # ∂²f/∂y²
        end
    end

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end

"""
Generate potential field and derivatives for f(x,y) = xy² + ln(x² + 1) + y³
"""
function generate_potential_2(nx, dx)
    # Grid generation
    x_range = range(-1, 1, length=nx)
    y_range = range(-1, 1, length=nx)

    μ = zeros(nx, nx)
    ∇μX = zeros(nx, nx)
    ∇μY = zeros(nx, nx)
    ∇2μX = zeros(nx, nx)
    ∇2μY = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            # Potential: f(x,y) = xy² + ln(x² + 1) + y³
            μ[i,j] = x*y^2 + log(x^2 + 1) + y^3

            # First derivatives
            ∇μX[i,j] = y^2 + 2*x/(x^2 + 1)    # ∂f/∂x
            ∇μY[i,j] = 2*x*y + 3*y^2          # ∂f/∂y

            # Second derivatives
            ∇2μX[i,j] = 2*(1 - x^2)/(x^2 + 1)^2  # ∂²f/∂x²
            ∇2μY[i,j] = 2*x + 6*y                # ∂²f/∂y²
        end
    end

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end

function generate_potential_3()
    # Generate potential field and derivatives
    μ = readdlm("pmf.in")

    # X-direction derivatives
    ∇μX = readdlm("damjux.in")
    ∇2μX = readdlm("d2amjux.in")

    # Y-direction derivatives
    ∇μY = readdlm("damjuy.in")
    ∇2μY = readdlm("d2amjuy.in")

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end
