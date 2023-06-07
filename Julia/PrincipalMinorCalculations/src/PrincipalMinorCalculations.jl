module PrincipalMinorCalculations

export mat2pm, pm2mat, pm_info_to_string, PMInfo

using LinearAlgebra

"""
Stores information and warnings about the principal minor computations.
"""
mutable struct PMInfo
    smallest_pivot::Float64
    number_of_times_ppivoted::Integer
    warn_not_odf::Bool
    warn_under_determined::Bool
    warn_inconsistent::Bool
end
PMInfo() = PMInfo(typemax(Float64), 0, false, false, false)

const PMDataType = ComplexF64   # The natural data type for principal minor computations

"""
mat2pm returns a 2^n - 1 vector of all the principal minors of the matrix A.
# Arguments
- `A::Matrix{T}`: n x n input matrix
- `thresh`::Float64`: Threshold for psuedo-pivoting.  Pseudo-pivoting will
  occur when a pivot smaller in magnitude than thresh arises.  Set thresh = 0
  to never pseudo-pivot except for a pivot of exactly zero.

Returns Tuple of (array of principal minors: Array, computation info: PMInfo).
    The structure of pm, where |A[v]| is the principal minor of "a" indexed by the vector v is:
    pm: |A[1]|, |A[2]|, |A[1 2]|, |A[3]|, |A[1 3]|, |A[2 3]|, |A[1 2 3]|,...
"""
function mat2pm(A, thresh=1e-5)
    @assert(length(size(A)) == 2 && size(A)[1] == size(A)[2])
    n = size(A)[1]
    scale = sum(abs.(A))/(n*n)  # average magnitude of matrix
    if scale == 0
        scale = 1               # prevent divide by 0 if matrix is zero
    end
    ppivot = scale              # value to use as a pivot if near 0 pivot arises

    zeropivs = []
    pm = Array{PMDataType}(undef, 2^n - 1)     # where the principal minors are stored
    ipm = 1                     # index for storing principal minors

    # q is the input queue of unprocessed matrices
    q = [A]                     # initial queue just has 1 matrix to process
    info = PMInfo()

    #
    # Main 'level' loop
    #
    for level ∈ 0:n-1
        nq = length(q)
        n1 = size(q[1], 1)

        # The output queue has twice the number of matrices, each one smaller
        # in row and col dimension
        qq = Vector{Matrix{PMDataType}}(undef, 2*nq)
        ipm1 = 1                # for indexing previous pm elements
        for i ∈ 1:nq
            a = q[i];
            pm[ipm] = a[1, 1];
            if n1 > 1
                abspiv = abs(pm[ipm]);
                if abspiv <= thresh
                    push!(zeropivs, ipm)
                    # Pivot nearly zero, use "pseudo-pivot"
                    pm[ipm] += ppivot;
                    abspiv = abs(pm[ipm]);
                end
                if abspiv < info.smallest_pivot
                    info.smallest_pivot = abspiv;
                end
                b = a[2:n1, 2:n1];
                d = a[2:n1, 1] ./ pm[ipm];
                c = b - d .* transpose(a[1, 2:n1]);

                # Order the output queue to make the elements of pm come out in
                # the correct order.
                qq[i] = b
                qq[i+nq] = c
            end

            if i > 1
                # if not the first iteration, to convert from a general pivot to a principal
                # minor, we need to multiply by every element of the pm matrix
                # we have already generated, in the order that we generated it.
                pm[ipm] = pm[ipm] * pm[ipm1]
                ipm1 += 1
            end
            ipm += 1
        end
        # Shallow copy for next outer loop.
        # qq will be reclaimed by the next qq = Vector{Matrix{}}() assignment.
        q = qq
    end

    #
    # Zero Pivot Loop
    #
    # Now correct principal minors for all places we used ppivot as a pivot
    # in place of a (near) 0.
    for i ∈ reverse(zeropivs)
        mask = i
        delta = msb(mask)
        delta2 = 2*delta
        ipm = (mask & ~delta)
        if ipm == 0
            pm[i] -= ppivot
        else
            pm[i] = (pm[i]/pm[ipm] - ppivot)*pm[ipm]
        end
        for j ∈ mask+delta2 : delta2 : length(pm)
            pm[j] -= ppivot*pm[j-delta]
        end
    end

    info.number_of_times_ppivoted = length(zeropivs)
    return pm, info
end

"""
pm2mat Finds a real or complex matrix that has pm as its principal minors.
pm2mat produces a matrix with some, but perhaps not all, of its
principal minors in common with pm.  If the principal minors are
not consistent or the matrix that has them is not ODF, pm2mat will
produce a matrix with different principal minors without throwing exceptions.
Run mat2pm on the output matrix a as needed.
# Arguments
- `pm::Array{T}`: The structure of pm, where |a[v]| is the principal minor of "a" indexed by the vector v is:
    pm: |a[1]|, |a[2]|, |a[1 2]|, |a[3]|, |a[1 3]|, |a[2 3]|, |a[1 2 3]|,...

:Returns tuple of (a::Matrix{T}, computation info: PMInfo)
"""
function pm2mat(input_pm)
    myeps = 1e-10
    n = UInt(round(log2(length(input_pm)+1)))

    # Make first (smallest) entry of zeropivs an impossible index
    zeropivs = [0]
    # Pick a random pseudo-pivot value that minimizes the chances that
    # pseudo-pivoting will create a non-ODF matrix.
    ppivot = 1.9501292851471754

    # To avoid division by zero, do an operation analogous to the zeropivot
    # loop in mat2pm.
    pm = Vector{PMDataType}(deepcopy(input_pm))  # Without the type conversion, the divides below can fail if inputs are all integers
    for i ∈ 1:(length(pm)+1) ÷ 2 - 1
        if abs(pm[i]) < myeps
            mask = i
            push!(zeropivs, i)
            ipm1 = mask & ~msb(mask)
            if ipm1 == 0
                pm[i] += ppivot
            else
                pm[i] = (pm[i]/pm[ipm1] + ppivot)*pm[ipm1]
            end
            delta = msb(mask)
            delta2 = 2*delta
            for j ∈ mask+delta2 : delta2 : length(pm)
                pm[j] += ppivot*pm[j - delta]
            end
        end
    end
    zeropivsidx = length(zeropivs) - 1
    zeropivsmax = zeropivs[zeropivsidx + 1]

    # initial processing is special, no call to invschurc is necessary
    nq = (length(pm)+1) ÷ 2
    q = Vector{Matrix{PMDataType}}(undef, nq)
    ipm1 = nq
    ipm2 = 1
    for i in 1:nq
        if i == 1
            pivot = pm[ipm1]
            q[i] = reshape([pivot], 1, 1)
        else
            pivot = pm[ipm1]/pm[ipm2]
            q[i] = reshape([pivot], 1, 1)
            ipm2 += 1
        end
        ipm1 += 1
    end

    #
    # Main 'level' loop
    #
    n1 = 1
    info = PMInfo()
    for level ∈ n-2:-1:0    # for consistency with mat2pm levels
        nq = length(q) ÷ 2
        n1 += 1
        # The output queue has half the number of matrices, each one larger in
        # row and col dimension
        qq = Vector{Matrix{PMDataType}}(undef, nq)

        ipm1 = 2*nq - 1
        ipm2 = nq - 1
        for i ∈ nq:-1:1     # process matrices in reverse order for zeropivs
            if i == 1
                pivot = pm[ipm1]
            else
                pivot = pm[ipm1]/pm[ipm2]
                ipm2 -= 1
            end
            info.smallest_pivot = min(info.smallest_pivot, abs(pivot))
            qq[i] = invschurc(info, pivot, q[i], q[i+nq])
            if zeropivsmax == ipm1
                qq[i][1, 1] -= ppivot
                zeropivsmax = zeropivs[zeropivsidx]
                zeropivsidx -= 1
            end
            ipm1 -= 1
        end
        # Shallow copy for next outer loop.
        # qq will be reclaimed by the next qq = Vector{Matrix{}}() assignment.
        q = qq
    end
    A = q[1]
    deskew!(A)
   return A, info
end

"""
Returns a human readable rendering (String) of the PMInfo structure.
"""
function pm_info_to_string(info::PMInfo)
    s = "Pseudo-pivoted $(info.number_of_times_ppivoted) times, smallest pivot used: $(info.smallest_pivot)"
    if info.warn_not_odf
        s *= ";  pm2mat: off diagonal zeros found, solution suspect."
    end
    if info.warn_under_determined
        s *= ";  pm2mat: multiple solutions to make rank(L-R)=1, solution suspect."
    end
    if info.warn_inconsistent
        s *= ";  pm2mat: input principal minors may be inconsistent, solution suspect."
    end
    return s
end

"""
Returns the numerical value of the most significant bit of x.
For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.
"""
function msb(x)
    if x == 0
        return 0
    end
    m = UInt(1)   # returns answer as UInt regardless of input type
    x >>= 1
    while x != 0
        m <<= 1
        x >>= 1
    end
    return m
end

"""
Suppose A is an (m+1) x (m+1) matrix such that

pivot = A[0, 0]
L = A[1:m+1, 1:m+1]
R = L - A[1:m+1, 0]*A[0, 1:m+1)/pivot = (the Schur's complement with
respect to the pivot or A/A[0]).

Then invschurc finds such an (m+1) x (m+1) matrix A (not necessarily
unique) given the pivot (a scalar), and the m x m matrices L and R.

If rank(L-R) is not 1, modifies R so that L-R is rank 1.

# Arguments
- `_pivot`: (1, 1) element of matrix a
- `L`: a[1:m+1, 1:m+1]
- `R`: L - a[1:m+1, 0]*a[0, 1:m+1)/_pivot

Returns Inverse Schur's complement
"""
function invschurc(info::PMInfo, pivot, L, R)
    # max allowable ratio of nearly zero row value to next largest row value
    myeps_i = 1e-3

    # make this relative to magnitude of R
    myeps_c = 1e-12 * norm(R, Inf)
    m = size(R)[1]

    # Try to make (L-R) rank 1
    if m == 2
        t1, t2 = solveright(info, L[1, 1], L[1, 2], L[2, 1], L[2, 2], R[1, 1], R[1, 2], R[2, 1], R[2, 2])

        # This is arbitrary, take the first.
        t = t1

        R[2, 1] = R[2, 1] * t
        R[1, 2] = R[1, 2] / t
    elseif m >= 3
        # We start with the lower right hand 3x3 submatrix.  We have 3
        # parameters, each with two possible solutions.  Only 1 of the 8
        # possible solutions need give us a L-R which is rank 1.  We find the
        # right solution by brute force.
        i1 = m - 2
        i2 = m - 1
        i3 = m
        r1, r2 = solveright(info, L[i2, i2], L[i2, i3], L[i3, i2], L[i3, i3], R[i2, i2], R[i2, i3], R[i3, i2], R[i3, i3])
        s1, s2 = solveright(info, L[i1, i1], L[i1, i2], L[i2, i1], L[i2, i2], R[i1, i1], R[i1, i2], R[i2, i1], R[i2, i2])
        t1, t2 = solveright(info, L[i1, i1], L[i1, i3], L[i3, i1], L[i3, i3], R[i1, i1], R[i1, i3], R[i3, i1], R[i3, i3])
        # Perform a parameterized "row reduction" on the first two rows of this
        # matrix and compute the absolute value of the (2,3) entry.  One of
        # them will be nearly zero.
        r111 = abs((L[i2, i3] - R[i2, i3] / r1) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s1) * (L[i1, i3] - R[i1, i3] / t1))
        r112 = abs((L[i2, i3] - R[i2, i3] / r1) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s1) * (L[i1, i3] - R[i1, i3] / t2))
        r121 = abs((L[i2, i3] - R[i2, i3] / r1) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s2) * (L[i1, i3] - R[i1, i3] / t1))
        r122 = abs((L[i2, i3] - R[i2, i3] / r1) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s2) * (L[i1, i3] - R[i1, i3] / t2))
        r211 = abs((L[i2, i3] - R[i2, i3] / r2) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s1) * (L[i1, i3] - R[i1, i3] / t1))
        r212 = abs((L[i2, i3] - R[i2, i3] / r2) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s1) * (L[i1, i3] - R[i1, i3] / t2))
        r221 = abs((L[i2, i3] - R[i2, i3] / r2) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s2) * (L[i1, i3] - R[i1, i3] / t1))
        r222 = abs((L[i2, i3] - R[i2, i3] / r2) * (L[i1, i1] - R[i1, i1]) -
                (L[i2, i1] - R[i2, i1] * s2) * (L[i1, i3] - R[i1, i3] / t2))
        rv = sort([r111, r112, r121, r122, r211, r212, r221, r222])
        mn = rv[1]
        if r111 == mn
            r, s, t = (r1, s1, t1)
        elseif r112 == mn
            r, s, t = (r1, s1, t2)
        elseif r121 == mn
            r, s, t = (r1, s2, t1)
        elseif r122 == mn
            r, s, t = (r1, s2, t2)
        elseif r211 == mn
            r, s, t = (r2, s1, t1)
        elseif r212 == mn
            r, s, t = (r2, s1, t2)
        elseif r221 == mn
            r, s, t = (r2, s2, t1)
        else  # (r222 == mn)
            r, s, t = (r2, s2, t2)
        end

        if mn > rv[2] * myeps_i
            info.warn_inconsistent = true
        end

        if sum(rv .< myeps_c) > 1
            info.warn_under_determined = true
        end

        R[i3, i2] = R[i3, i2] * r
        R[i2, i3] = R[i2, i3] / r
        R[i2, i1] = R[i2, i1] * s
        R[i1, i2] = R[i1, i2] / s
        R[i3, i1] = R[i3, i1] * t
        R[i1, i3] = R[i1, i3] / t

        # Now the lower right hand 3x3 submatrix of L-R has rank 1.  Then we
        # fix up the rest of L-R.
        for i1 in m-3 : -1 : 1
            i2 = i1 + 1
            i3 = i1 + 2

            # Now the inside lower right submatrix is done, so we
            # only have 2 free parameters and 4 combinations to examine.
            s1, s2 = solveright(info, L[i1, i1], L[i1, i2], L[i2, i1], L[i2, i2], R[i1, i1], R[i1, i2],
                                R[i2, i1], R[i2, i2])
            t1, t2 = solveright(info, L[i1, i1], L[i1, i3], L[i3, i1], L[i3, i3], R[i1, i1], R[i1, i3],
                                R[i3, i1], R[i3, i3])

            r11 = abs((L[i2, i3] - R[i2, i3]) * (L[i1, i1] - R[i1, i1]) -
                    (L[i2, i1] - R[i2, i1] * s1) * (L[i1, i3] - R[i1, i3] / t1))
            r12 = abs((L[i2, i3] - R[i2, i3]) * (L[i1, i1] - R[i1, i1]) -
                    (L[i2, i1] - R[i2, i1] * s1) * (L[i1, i3] - R[i1, i3] / t2))
            r21 = abs((L[i2, i3] - R[i2, i3]) * (L[i1, i1] - R[i1, i1]) -
                    (L[i2, i1] - R[i2, i1] * s2) * (L[i1, i3] - R[i1, i3] / t1))
            r22 = abs((L[i2, i3] - R[i2, i3]) * (L[i1, i1] - R[i1, i1]) -
                    (L[i2, i1] - R[i2, i1] * s2) * (L[i1, i3] - R[i1, i3] / t2))
            rv = sort([r11, r12, r21, r22])
            mn = rv[1]
            if r11 == mn
                s, t = (s1, t1)
            elseif r12 == mn
                s, t = (s1, t2)
            elseif r21 == mn
                s, t = (s2, t1)
            else  # (r22 == mn)
                s, t = (s2, t2)
            end

            if mn > rv[2] * myeps_i
                info.warn_inconsistent = true
            end

            if sum(rv .< myeps_c) > 1
                info.warn_under_determined = true
            end

            R[i2, i1] = R[i2, i1] * s
            R[i1, i2] = R[i1, i2] / s
            R[i3, i1] = R[i3, i1] * t
            R[i1, i3] = R[i1, i3] / t
            for j in i1 + 3 : m
                # Finally, once the second row of the submatrix we are working
                # on is uniquely solved, we just pick the solution to the
                # quadratic such that the the first row is a multiple of the
                # second row.  Note that one of r1, r2 will be almost zero.
                # Solving the quadratics leads to much better performance
                # numerically than just taking multiples of the second or
                # any other row.
                #

                j1 = i1 + 1
                t1, t2 = solveright(info, L[i1, i1], L[i1, j], L[j, i1], L[j, j], R[i1, i1], R[i1, j],
                                    R[j, i1], R[j, j])
                r1 = abs((L[j1, j] - R[j1, j]) * (L[i1, i1] - R[i1, i1]) -
                        (L[j1, i1] - R[j1, i1]) * (L[i1, j] - R[i1, j] / t1))
                r2 = abs((L[j1, j] - R[j1, j]) * (L[i1, i1] - R[i1, i1]) -
                        (L[j1, i1] - R[j1, i1]) * (L[i1, j] - R[i1, j] / t2))
                if r1 <= r2
                    t = t1
                else
                    t = t2
                end

                rv = sort([r1, r2])
                mn = rv[1]
                if mn > rv[2] * myeps_i
                    info.warn_inconsistent = true
                end

                if sum(rv .< myeps_c) > 1
                    info.warn_under_determined = true
                end

                R[j, i1] = R[j, i1] * t
                R[i1, j] = R[i1, j] / t
            end
        end
    end

    B = (L - R)  # B is a rank 1 matrix
    idxmax = argmax(abs.(diag(B)))
    # For numerical reasons use the largest diagonal element as a base to find
    # the two vectors whose outer product is B*pivot
    y_t = B[idxmax, :]
    if y_t[idxmax] == 0
        # This shouldn't happen normally, but to prevent
        # divide by zero when we set all "dependent" principal
        # minors (with index sets greater than or equal to a constant)
        # to the same value, let yT be something.
        y_t = ones(m)
    end

    x = B[:, idxmax] * pivot / y_t[idxmax]
    A = zeros(PMDataType, m + 1, m + 1)
    A[1, 1] = pivot
    A[1, 2:m + 1] = y_t
    A[2:m + 1, 1] = x
    A[2:m + 1, 2:m + 1] = L
    return A
end

"""
Returns the two possible real (when the principal minors come from an ODF matrix)
solutions that will make L-R rank one if we let
r21 = r21*t? (where ? = 1 or 2) and
r12 = r12/t?

Returns Tuple of real solutions
"""
function solveright(info::PMInfo, l11, l12, l21, l22, r11, r12, r21, r22)
    x1 = l11 - r11
    x2 = l22 - r22
    d = sqrt(x1*x1*x2*x2 + l12*l12*l21*l21 + r12*r12*r21*r21 - 2*x1*x2*l12*l21 -
    2*x1*x2*r12*r21 - 2*l12*l21*r21*r12)
    if l12 == 0 || r21 == 0
        # This shouldn't happen normally, but to prevent
        # divide by zero when we set all "dependent" principal
        # minors (with index sets greater than or equal to a constant)
        # to the same value, let [t1,t2] be something.
        t1 = 1
        t2 = 1
        info.warn_not_odf = true
    else
        t1 = (-x1*x2 + l12*l21 + r12*r21 - d) / (2*l12*r21)
        t2 = (-x1*x2 + l12*l21 + r12*r21 + d) / (2*l12*r21)

        # This also shouldn't happen.  Comment above applies.
        if t1 == 0 || t2 == 0
            info.warn_not_odf = true
            if t1 == 0 && t2 == 0
                t1 = 1
                t2 = 1
            elseif t1 == 0
                # return better solution in t1 for m=2 case in invschurc
                t1 = t2
                t2 = 1
            else  # t2 == 0
                t2 = 1
            end
        end
    end
    return t1, t2
end

"""
Makes abs(a[0, i]) = abs(a[i, 0]) through diagonal similarity for all i.
# Arguments
- `A::Matrix{T}`: Square matrix

Returns deskewed square matrix (but also modifies A by reference)
"""
function deskew!(A)
    @assert(length(size(A)) == 2 && size(A)[1] == size(A)[2])
    n = size(A)[1]
    d = ones(Float64, n)
    for i ∈ 2:n
        if A[i, 1] != 0  # don't divide by 0
            d[i] = sqrt(abs(A[1, i]/A[i, 1]))
            if d[i] > 1e6 || d[i] < 1e-6
                # something is likely wrong, use 1 instead
                d[i] = 1.
            end
        end     # else leave d(i) = 1
    end

    # If D = diag(d), this effectively computes A = D*A*inv(D)
    for i ∈ 2:n
        A[i, :] *= d[i]
    end
    for i ∈ 2:n
        A[:, i] /= d[i]
    end
    return A
end

end # module PrincipalMinorCalculations
