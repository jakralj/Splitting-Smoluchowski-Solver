using Plots, LinearAlgebra, Statistics
include("../../methods.jl")
include("../../utils/Potentials/potentials.jl")

#https://scicomp.stackexchange.com/questions/19749/correct-way-of-computing-norm-l-2-for-a-finite-difference-scheme
l2(u1, u2, dx) = sqrt(sum((u1 .- u2).^2)*dx)


