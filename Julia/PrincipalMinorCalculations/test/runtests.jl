module TestPrincipalMinorCalculations

using Test
# using Profile
using PrincipalMinorCalculations

@testset "msb" begin
    @test PrincipalMinorCalculations.msb(1) == 1
    @test PrincipalMinorCalculations.msb(2) == 2
    @test PrincipalMinorCalculations.msb(7) == 4
    @test PrincipalMinorCalculations.msb(6) == 4
    @test PrincipalMinorCalculations.msb(13) == 8
    @test PrincipalMinorCalculations.msb(0x1234567812345678) == 0x1000000000000000
    @test PrincipalMinorCalculations.msb(0xf234567812345678) == 0x8000000000000000
end

@testset "principal minor calculations" begin
    m1 = [  1 2;
            3 4]
    pm1, info = mat2pm(m1)
    @test pm1 == [1, 4, -2]
    m2, info = pm2mat(pm1)
    pm2, info = mat2pm(m2)
    @test pm1 == pm2

    m1 = [  1 2 6;
            2 4 5;
           -1 2 3]
    pm1, info = mat2pm(m1)
    @test pm1 == [1, 4, 0, 3, 9, 2, 28]
    m2, info = pm2mat(pm1)
    pm2, info = mat2pm(m2)
    @test pm1 ≈ pm2 atol=1e-10  # 1e-14 passes, but put in some leeway

    m1 = [  0 1 0 0;
            0 0 1 0;
            0 0 0 1;
            1 0 0 0]
    pm1, info = mat2pm(m1)
    @test pm1 == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
    @test info.number_of_times_ppivoted == 7
    @test info.smallest_pivot == 0.25
    m2, info = pm2mat(pm1)
    @test info.number_of_times_ppivoted == 0
    @test info.smallest_pivot ≈ 1.9501292851471754 atol=1e-10
    @test m2 == [   0 1 1 1;
                    0 0 1 1;
                    0 0 0 1;
                    1 0 0 0]
    @test info.warn_not_odf
    @test info.warn_under_determined
    @test !info.warn_inconsistent

    pm_inconsistent = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    m, info = pm2mat(pm_inconsistent)
    @test !info.warn_not_odf
    @test !info.warn_under_determined
    @test info.warn_inconsistent
    pm, info = mat2pm(m)
    # all but two of the principal minors actually match
    pm[13] = 6.
    pm[15] = 8.
    @test pm ≈ pm_inconsistent atol=1e-10

    m1 = [  -6 3 -9 4;
            -6 -5 3 6;
            3 -3 6 -7;
            1 1 -1 -3]
    pm1, info = mat2pm(m1)
    pm_truth = [-6, -5, 48, 6, -9, -21, -36, -3, 14, 9, -94, -25, 96, 59, 6]
    @test info.number_of_times_ppivoted == 0
    @test info.smallest_pivot == 0.75
    @test pm1 ≈ pm_truth atol=1e-10
    m2, info = pm2mat(pm_truth)
    pm2, info = mat2pm(m2)
    @test pm1 ≈ pm2 atol=1e-10

    m1 = [  1 + 1im  3 - 4im -2 - 1im;
            7 + 3im -5 + 2im -1 - 1im;
            0 - 4im  3 + 0im  3 + 3im]
    pm1, info = mat2pm(m1)
    pm_truth = [1.0 + 1.0im, -5.0 + 2.0im, -40.0 + 16.0im, 3.0 + 3.0im, 4.0 - 2.0im, -18.0 - 6.0im, -201.0 - 29.0im]
    @test info.number_of_times_ppivoted == 0
    @test info.smallest_pivot == √2
    @test pm1 ≈ pm_truth atol=1e-10
    m2, info = pm2mat(pm_truth)
    pm2, info = mat2pm(m2)
    @test pm1 ≈ pm2 atol=1e-10

    n = 18
    m1 = rand(Float64, (n, n))
    pm1, info = mat2pm(m1)
    # @time mat2pm(m1)

    m2, info = pm2mat(pm1)
    # @time pm2mat(pm1)
    # @profile pm2mat(pm1)
    # Profile.print(C = true, noisefloor = 2.0)
    # readline()

    pm2, info = mat2pm(m2)
    @test length(pm1) == 2^n - 1
    @test length(pm2) == 2^n - 1
    @test pm1 ≈ pm2 atol=1e-7
end

@testset "pm_info_to_string" begin
        info = PMInfo()
        info.smallest_pivot = 3.14
        info.number_of_times_ppivoted = 42
        s = pm_info_to_string(info)
        @test s == "Pseudo-pivoted 42 times, smallest pivot used: 3.14"
        info.warn_not_odf = true
        info.warn_under_determined = true
        info.warn_inconsistent = true
        s = pm_info_to_string(info)
        @test s == "\
            Pseudo-pivoted 42 times, smallest pivot used: 3.14;  \
            pm2mat: off diagonal zeros found, solution suspect.;  \
            pm2mat: multiple solutions to make rank(L-R)=1, solution suspect.;  \
            pm2mat: input principal minors may be inconsistent, solution suspect."
end

end # module TestPrincipalMinorCalculations
