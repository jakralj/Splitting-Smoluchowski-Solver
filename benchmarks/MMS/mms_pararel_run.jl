using Distributed

@everywhere include("mms_distributed.jl")

dt_values = 2 .^(-1.0 .*(6:16))
dx_values = 2 .^(-1.0 .*(3:8))

run_mms_verification_ultra_parallel(mms1, dx_values, dt_values, "mms1.tsv")
run_mms_verification_ultra_parallel(mms2, dx_values, dt_values, "mms2.tsv")
