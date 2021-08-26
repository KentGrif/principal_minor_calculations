import numpy as np


# MAT2PM Finds principal minors of an n x n real or complex matrix.
#   PM = MAT2PM(A)
#   where "A" is an n x n matrix in which zero can arise as a pivot at any
#   point.  MAT2PM returns a 2^n - 1 vector of all the principal minors
#   of the matrix "A".
#
#   PM = MAT2PM(A, THRESH)
#   Explicitly sets the pseudo-pivot threshold to THRESH.  Pseudo-pivoting
#   will occur when a pivot smaller in magnitude than THRESH arises.  Set
#   THRESH = 0 to never pseudo-pivot except for a pivot of exactly zero.
#
#   The structure of PM, where |A[v]| is the principal minor of "A" indexed
#   by the vector v:
#   PM: |A[1]|, |A[2]|, |A[1 2]|, |A[3]|, |A[1 3]|, |A[2 3]|, |A[1 2 3]|,...
def mat2pm(a, thresh=1e-5):
    # Only works on up to 48x48 matrices due to restrictions
    # on bitcmp and indices.
    assert(len(a.shape) == 2 and a.shape[0] == a.shape[1])
    n = a.shape[0]
    scale = np.sum(abs(a))/(n*n)    # average magnitude of matrix
    if scale == 0:
        scale = 1               # prevent divide by 0 if matrix is zero
    ppivot = scale              # value to use as a pivot if near 0 pivot arises

    zeropivs = []
    pm = np.zeros(2**n - 1)     # where the principal minors are stored
    ipm = 0                     # index for storing principal minors

    # q is the input queue of unprocessed matrices
    q = [a]                     # initial queue just has 1 matrix to process
    pivmin = np.inf             # keep track of smallest pivot actually used

    #
    # Main 'level' loop
    #
    for level in range(n):
        nq = len(q)
        n1 = q[0].shape[0]
        # The output queue has twice the number of matrices, each one smaller
        # in row and col dimension
        qq = [None] * 2*nq      # zeros(n1-1, n1-1, nq*2);
        ipm1 = 0                # for indexing previous pm elements
        for i in range(nq):
            a = q[i]
            pm[ipm] = a[0, 0]
            if n1 > 1:
                abspiv = np.abs(pm[ipm])
                if abspiv <= thresh:
                    zeropivs.append(ipm)
                    # Pivot nearly zero, use "pseudo-pivot"
                    pm[ipm] += ppivot
                    abspiv = np.abs(pm[ipm])
                if abspiv < pivmin:
                    pivmin = abspiv
                b = a[1:, 1:].copy()            # a(2:n1,2:n1);
                d = a[1:, 0] / pm[ipm]          # a(2:n1,1)/pm(ipm);
                c = b - np.outer(d, a[0, 1:])   # a(1,2:n1);

                # Order the output queue to make the elements of pm come out in
                # the correct order.
                qq[i] = b
                qq[i+nq] = c
            if i > 0:
                # if i > 1, to convert from a general pivot to a principal
                # minor, we need to multiply by every element of the pm matrix
                # we have already generated, in the order that we generated it.
                pm[ipm] = pm[ipm]*pm[ipm1]
                ipm1 += 1
            ipm += 1
        q = qq

    #
    # Zero Pivot Loop
    #
    # Now correct principal minors for all places we used ppivot as a pivot
    # in place of a (near) 0.
    for i in reversed(zeropivs):
        mask = i + 1
        delta = _msb(mask)
        delta2 = 2*delta
        ipm1 = mask & ~delta
        if ipm1 == 0:
            pm[mask-1] = pm[mask-1] - ppivot
        else:
            pm[mask-1] = (pm[mask-1]/pm[ipm1-1] - ppivot)*pm[ipm1-1]
        for j in range(mask+delta2, 2**n, delta2):
            pm[j-1] = pm[j-1] - ppivot*pm[j - delta - 1]
    # Warn user in case larger pivots are desired
    print(f'MAT2PM: pseudo-pivoted {len(zeropivs)} times, smallest pivot used: {pivmin}')
    return pm

# Returns the numerical value of the most significant bit of x.
# For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.
# function [m] = msb(x)
# persistent MSBTABLE     # MSBTABLE persists between calls to mat2pm
# if isempty(MSBTABLE)
#     # If table is empty, initialize it
#     MSBTABLE = zeros(255,1);
#     for i=1:255
#         MSBTABLE(i) = msbslow(i);
#     end
# end
#
# m = 0;
# # process 8 bits at a time for speed
# if x ~= 0
#     while x ~= 0
#         x1 = x;
#         x = bitshift(x, -8);    # 8 bit left shift
#         m = m + 8;
#     end
#     m = bitshift(MSBTABLE(x1), m-8); # right shift
# end


# Returns the numerical value of the most significant bit of x.
# For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.  Slow version
# used to build a table.
def _msb(x):
    assert(x > 0)
    m = 1
    x >>= 1
    while x != 0:
        m <<= 1
        x >>= 1
    return m
