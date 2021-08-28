import numpy as np


def mat2pm(a: np.array, thresh=1e-5):
    """
    mat2pm returns a 2^n - 1 vector of all the principal minors of the matrix a.
    :param a: n x n input matrix
    :param thresh: Threshold for psuedo-pivoting.  Pseudo-pivoting will occur when a pivot smaller in magnitude than
        thresh arises.  Set thresh = 0 to never pseudo-pivot except for a pivot of exactly zero.
    :return: np.array of principal minors.  The structure of pm, where |a[v]| is the principal minor of "a" indexed
        by the vector v is:
        pm: |a[1]|, |a[2]|, |a[1 2]|, |a[3]|, |a[1 3]|, |a[2 3]|, |a[1 2 3]|,...
    """
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
                b = a[1:, 1:]                   # a(2:n1,2:n1);
                d = a[1:, 0] / pm[ipm]          # a(2:n1,1)/pm(ipm);
                c = b - np.outer(d, a[0, 1:])   # a(1,2:n1);

                # Order the output queue to make the elements of pm come out in
                # the correct order.
                qq[i] = b
                qq[i+nq] = c
            if i > 0:
                # if not the first iteration, to convert from a general pivot to a principal
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
        ipm = (mask & ~delta)
        if ipm == 0:
            pm[i] = pm[i] - ppivot
        else:
            ipm -= 1
            pm[i] = (pm[i]/pm[ipm] - ppivot)*pm[ipm]
        for j in range(mask+delta2-1, 2**n-1, delta2):
            pm[j] = pm[j] - ppivot*pm[j-delta]

    # print(f'mat2pm: pseudo-pivoted {len(zeropivs)} times, smallest pivot used: {pivmin}')
    return pm


# Returns the numerical value of the most significant bit of x.
# For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.
def _msb(x: int):
    if x == 0:
        return 0
    m = 1
    x >>= 1
    while x != 0:
        m <<= 1
        x >>= 1
    return m

# Doesn't seem to help much in Python, probably throw this away
# _msbtable = None
#
#
# # Returns the numerical value of the most significant bit of x.
# # For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.
# def _msb_fast(x: int):
#     global _msbtable
#     if not _msbtable:
#         # If table is empty, initialize it
#         _msbtable = []
#         for i in range(256):
#             _msbtable.append(_msb(i))
#
#     m = 0
#     # process 8 bits at a time for speed
#     while x != 0:
#         x1 = x
#         x >>= 8
#         m += 8
#     m = _msbtable[x1] << (m-8)
#     return m
