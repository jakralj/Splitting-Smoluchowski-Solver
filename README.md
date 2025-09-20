# GPU paralelized splitting methods for numerical solving of Smulochowski diffusion equation 

This code accompanies the article "Numerical Solving of Smulochowski Diffusion Equation in Two Dimensions". It uses Lie and Strang splitting to compute fast and accurate numerical solutions of Smulochowski diffusion equation. For comparison alternating direction implicit method and Euler forward difference are also implemented, however, based on much inferior performance, we actively discourage their use. 

Code is written in Julia, and uses several external libraries, that must be installed. The GPU portion of the code also depends on CUDA. File `methods.jl` contains sequential implementation of these 4 methods, `gpu.jl` implements paralelised CUDA implementations. Folder `utils` contains 2 scripts - Hex Placement implements a way to generate potentials for nerve crossections, used as an example of a realistic potentials in the article. `Potentials` contain code for parsing the output files of `Hex placement` and also defines 2 analytic test potentials. 

Folder `Benchmark` contains scripts used to create data and figures for the article. 
