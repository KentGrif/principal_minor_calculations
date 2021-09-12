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
            pm[i] -= ppivot
        else:
            ipm -= 1
            pm[i] = (pm[i]/pm[ipm] - ppivot)*pm[ipm]
        for j in range(mask+delta2-1, 2**n-1, delta2):
            pm[j] -= ppivot*pm[j-delta]

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


# PM2MAT Finds a real or complex matrix that has PM as its principal
# minors.
#
#   PM2MAT produces a matrix with some, but perhaps not all, of its
#   principal minors in common with PM.  If the principal minors are
#   not consistent or the matrix that has them is not ODF, PM2MAT will
#   produce a matrix with different principal minors without warning.
#   Run MAT2PM on the output matrix A as needed.
#
#   A = PM2MAT(PM)
#   where PM is a 2^n - 1 vector of principal minors and A will be an n x n
#   matrix.
#
#   The structure of PM, where |A[v]| is the principal minor of "A" indexed
#   by the vector v:
#   PM: |A[1]| |A[2]| |A[1 2]| |A[3]| |A[1 3]| |A[2 3]| |A[1 2 3]| ...
def pm2mat(pm):
    myeps = 1e-10

    n = int(round(np.log2(len(pm)+1)))

    # Make first (smallest) entry of zeropivs an impossible index
    zeropivs = [-1]
    # Pick a random pseudo-pivot value that minimizes the chances that
    # pseudo-pivoting will create a non-ODF matrix.
    ppivot = 1.9501292851471754

    # initialize globals to allow warnings to be printed only once
    # global WARN_A WARN_C WARN_I
    # WARN_A = false; WARN_C = false; WARN_I = false;

    # To avoid division by zero, do an operation analogous to the zeropivot
    # loop in mat2pm.
    for i in range((len(pm)+1)//2):        # 1:((length(pm)+1)/2 - 1)
        if np.abs(pm[i]) < myeps:
            mask = i + 1
            zeropivs.append(i)
            ipm1 = mask & ~_msb(mask)
            if ipm1 == 0:
                pm[i] += ppivot
            else:
                pm[i] = (pm[i]/pm[ipm1-1] + ppivot)*pm[ipm1-1]
            delta = _msb(mask)
            delta2 = 2*delta
            for j in range(mask+delta2-1, 2**n-1, delta2):
                pm[j] += ppivot*pm[j - delta]
    zeropivsidx = len(zeropivs) - 1
    zeropivsmax = zeropivs[zeropivsidx]
    zeropivsidx -= 1

    # initial processing is special, no call to invschurc is necessary
    nq = 2**(n-1)
    n1 = 1
    q = [None] * nq   # zeros(1,1,nq)
    ipm1 = nq-1
    ipm2 = 0
    for i in range(nq):
        if i == 0:
            q[i] = np.array([[pm[ipm1]]])
        else:
            q[i] = np.array([[pm[ipm1]/pm[ipm2]]])
            ipm2 += 1
        ipm1 += 1

    #
    # Main 'level' loop
    #
    for level in range(n-2, -1, -1):  # just counts? = n-2:-1:0        # for consistency with mat2pm levels
        nq = len(q) // 2
        n1 = n1+1
        # The output queue has half the number of matrices, each one larger in
        # row and col dimension
        qq = [None] * nq        # zeros(n1, n1, nq);

        ipm1 = 2*nq-2
        ipm2 = nq-2
        for i in range(nq-1, -1, -1):  # = nq:-1:1     # process matrices in reverse order for zeropivs
            if i == 0:
                pivot = pm[ipm1]
            else:
                pivot = pm[ipm1]/pm[ipm2]
                ipm2 -= 1
            qq[i] = _invschurc(pivot, q[i], q[i+nq])
            if zeropivsmax == ipm1:
                qq[i][0][0] -= ppivot
                zeropivsmax = zeropivs[zeropivsidx]
                zeropivsidx -= zeropivsidx
            ipm1 -= 1
        q = qq
    a = q[0]
    # a = _deskew(a)
    return a
    # if WARN_A
    #     # ODF (a) not satisfied
    #     fprintf(2,...
    # 'PM2MAT: off diagonal zeros found, solution suspect.\n');
    # end
    # if WARN_C
    #     fprintf(2,...
    # 'PM2MAT: multiple solutions to make rank(L-R)=1, solution suspect.\n');
    # end
    # if WARN_I
    #     fprintf(2, ...
    # 'PM2MAT: input principal minors may be inconsistent, solution suspect.\n');
    # end


#
# Suppose A is an (m+1) x (m+1) matrix such that
#
# pivot = A(1,1)
# L = A(2:m+1, 2:m+1)
# R = L - A(2:m+1,1)*A(1,2:m+1)/pivot = (the Schur's complement with
# respect to the pivot or A/A[1]).
#
# Then invschurc finds such an (m+1) x (m+1) matrix A (not necessarily
# unique) given the pivot (a scalar), and the m x m matrices L and R.
#
# If rank(L-R) is not 1, modifies R so that L-R is rank 1.
#
def _invschurc(pivot, L, R):
    # global WARN_C WARN_I
    myeps_i = 1e-3 * np.linalg.norm(R, ord=np.inf)   # make these relative to magnitude of R
    myeps_c = 1e-9 * np.linalg.norm(R, ord=np.inf)
    m = R.shape[0]

    # Try to make (L-R) rank 1
    if m == 2:
        [t1, t2] = _solveright(L[0, 0], L[0, 1], L[1, 0], L[1, 1], R[0, 0], R[0, 1], R[1, 0], R[1, 1])

        # This is arbitrary, take the first.
        t = t1

        R[1, 0] = R[1, 0]*t
        R[0, 1] = R[0, 1]/t
    elif m >= 3:
        # We start with the lower right hand 3x3 submatrix.  We have 3
        # parameters, each with two possible solutions.  Only 1 of the 8
        # possible solutions need give us a L-R which is rank 1.  We find the
        # right solution by brute force.
        i1 = m-3
        i2 = m-2
        i3 = m-1
        [r1, r2] = _solveright(L[i2, i2], L[i2, i3], L[i3, i2], L[i3, i3], R[i2, i2], R[i2, i3], R[i3, i2], R[i3, i3])
        [s1, s2] = _solveright(L[i1, i1], L[i1, i2], L[i2, i1], L[i2, i2], R[i1, i1], R[i1, i2], R[i2, i1], R[i2, i2])
        [t1, t2] = _solveright(L[i1, i1], L[i1, i3], L[i3, i1], L[i3, i3], R[i1, i1], R[i1, i3], R[i3, i1], R[i3, i3])
        # Perform a parameterized "row reduction" on the first two rows of this
        # matrix and compute the absolute value of the (2,3) entry.  One of
        # them will be nearly zero.
        r111 = abs((L[i2, i3] - R[i2, i3]/r1)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s1)*(L[i1, i3] - R[i1, i3]/t1))
        r112 = abs((L[i2, i3] - R[i2, i3]/r1)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s1)*(L[i1, i3] - R[i1, i3]/t2))
        r121 = abs((L[i2, i3] - R[i2, i3]/r1)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s2)*(L[i1, i3] - R[i1, i3]/t1))
        r122 = abs((L[i2, i3] - R[i2, i3]/r1)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s2)*(L[i1, i3] - R[i1, i3]/t2))
        r211 = abs((L[i2, i3] - R[i2, i3]/r2)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s1)*(L[i1, i3] - R[i1, i3]/t1))
        r212 = abs((L[i2, i3] - R[i2, i3]/r2)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s1)*(L[i1, i3] - R[i1, i3]/t2))
        r221 = abs((L[i2, i3] - R[i2, i3]/r2)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s2)*(L[i1, i3] - R[i1, i3]/t1))
        r222 = abs((L[i2, i3] - R[i2, i3]/r2)*(L[i1, i1] - R[i1, i1]) -
                   (L[i2, i1] - R[i2, i1]*s2)*(L[i1, i3] - R[i1, i3]/t2))
        rv = [r111, r112, r121, r122, r211, r212, r221, r222]
        mn = np.amin(rv)
        if r111 == mn:
            [r, s, t] = [r1, s1, t1]
        elif r112 == mn:
            [r, s, t] = [r1, s1, t2]
        elif r121 == mn:
            [r, s, t] = [r1, s2, t1]
        elif r122 == mn:
            [r, s, t] = [r1, s2, t2]
        elif r211 == mn:
            [r, s, t] = [r2, s1, t1]
        elif r212 == mn:
            [r, s, t] = [r2, s1, t2]
        elif r221 == mn:
            [r, s, t] = [r2, s2, t1]
        else:  # (r222 == mn)
            [r, s, t] = [r2, s2, t2]

        if mn > myeps_i:
            pass
            # WARN_I = true;

        if np.count_nonzero(rv < myeps_c) > 1:
            pass
            # WARN_C = true;

        R[i3, i2] = R[i3, i2]*r
        R[i2, i3] = R[i2, i3]/r
        R[i2, i1] = R[i2, i1]*s
        R[i1, i2] = R[i1, i2]/s
        R[i3, i1] = R[i3, i1]*t
        R[i1, i3] = R[i1, i3]/t

        # Now the lower right hand 3x3 submatrix of L-R has rank 1.  Then we
        # fix up the rest of L-R.
        for i1 in range(m-4, -1, -1):
            i2 = i1+1
            i3 = i1+2

            # Now the inside lower right submatrix is done, so we
            # only have 2 free parameters and 4 combinations to examine.
            [s1, s2] = _solveright(L[i1, i1], L[i1, i2], L[i2, i1], L[i2, i2], R[i1, i1], R[i1, i2], R[i2, i1],
                                   R[i2, i2])
            [t1, t2] = _solveright(L[i1, i1], L[i1, i3], L[i3, i1], L[i3, i3], R[i1, i1], R[i1, i3], R[i3, i1],
                                   R[i3, i3])

            r11 = abs((L[i2, i3] - R[i2, i3])*(L[i1, i1] - R[i1, i1]) -
                      (L[i2, i1] - R[i2, i1]*s1)*(L[i1, i3] - R[i1, i3]/t1))
            r12 = abs((L[i2, i3] - R[i2, i3])*(L[i1, i1] - R[i1, i1]) -
                      (L[i2, i1] - R[i2, i1]*s1)*(L[i1, i3] - R[i1, i3]/t2))
            r21 = abs((L[i2, i3] - R[i2, i3])*(L[i1, i1] - R[i1, i1]) -
                      (L[i2, i1] - R[i2, i1]*s2)*(L[i1, i3] - R[i1, i3]/t1))
            r22 = abs((L[i2, i3] - R[i2, i3])*(L[i1, i1] - R[i1, i1]) -
                      (L[i2, i1] - R[i2, i1]*s2)*(L[i1, i3] - R[i1, i3]/t2))
            rv = [r11, r12, r21, r22]
            mn = np.amin(rv)
            if r11 == mn:
                [s, t] = [s1, t1]
            elif r12 == mn:
                [s, t] = [s1, t2]
            elif r21 == mn:
                [s, t] = [s2, t1]
            else:  # (r22 == mn)
                [s, t] = [s2, t2]

            if mn > myeps_i:
                pass
                # WARN_I = true;

            if np.count_nonzero(rv < myeps_c) > 1:
                pass
                # WARN_C = true;

            R[i2, i1] = R[i2, i1]*s
            R[i1, i2] = R[i1, i2]/s
            R[i3, i1] = R[i3, i1]*t
            R[i1, i3] = R[i1, i3]/t
            for j in range(i1+2, m):
                # Finally, once the second row of the submatrix we are working
                # on is uniquely solved, we just pick the solution to the
                # quadratic such that the the first row is a multiple of the
                # second row.  Note that one of r1, r2 will be almost zero.
                # Solving the quadratics leads to much better performance
                # numerically than just taking multiples of the second or
                # any other row.
                #

                j1 = i1+1
                [t1, t2] = _solveright(L[i1, i1], L[i1, j], L[j, i1], L[j, j], R[i1, i1], R[i1, j], R[j, i1], R[j, j])
                r1 = abs((L[j1, j] - R[j1, j])*(L[i1, i1] - R[i1, i1]) -
                         (L[j1, i1] - R[j1, i1])*(L[i1, j] - R[i1, j]/t1))
                r2 = abs((L[j1, j] - R[j1, j])*(L[i1, i1] - R[i1, i1]) -
                         (L[j1, i1] - R[j1, i1])*(L[i1, j] - R[i1, j]/t2))
                if r1 <= r2:
                    t = t1
                else:
                    t = t2

                rv = [r1, r2]
                if mn > myeps_i:
                    pass
                    # WARN_I = true;

                if np.count_nonzero(rv < myeps_c) > 1:
                    pass
                    # WARN_C = true;

                R[j, i1] = R[j, i1]*t
                R[i1, j] = R[i1, j]/t

    B = (L-R)    # B is a rank 1 matrix
    idxmax = np.argmax(abs(np.diagonal(B)))
    # For numerical reasons use the largest diagonal element as a base to find
    # the two vectors whose outer product is B*pivot
    yT = B[idxmax, :]
    if yT[idxmax] == 0:
        # This shouldn't happen normally, but to prevent
        # divide by zero when we set all "dependent" principal
        # minors (with index sets greater than or equal to a constant)
        # to the same value, let yT be something.
        yT = np.ones(m)

    x = B[:, idxmax]*pivot / yT[idxmax]
    A = np.zeros((m+1, m+1))
    A[0, 0] = pivot
    A[0, 1:m+1] = yT
    A[1:m+1, 0] = x
    A[1:m+1, 1:m+1] = L
    return A


#
# Returns the two possible real solutions that will make L-R rank one if we
# let
# r21 = r21*t? (where ? = 1 or 2) and
# r12 = r12/t?
#
def _solveright(l11, l12, l21, l22, r11, r12, r21, r22):
    # global WARN_A
    x1 = l11-r11
    x2 = l22-r22
    d = np.sqrt(x1*x1*x2*x2 + l12*l12*l21*l21 + r12*r12*r21*r21 - 2*x1*x2*l12*l21 - 2*x1*x2*r12*r21-2*l12*l21*r21*r12)
    if (l12 == 0) or (r21 == 0):
        # This shouldn't happen normally, but to prevent
        # divide by zero when we set all "dependent" principal
        # minors (with index sets greater than or equal to a constant)
        # to the same value, let [t1,t2] be something.
        t1 = 1
        t2 = 1
        # WARN_A = true;
    else:
        t1 = (-x1*x2 + l12*l21 + r12*r21 - d)/(2*l12*r21)
        t2 = (-x1*x2 + l12*l21 + r12*r21 + d)/(2*l12*r21)

        # This also shouldn't happen.  Comment above applies.
        if (t1 == 0) or (t2 == 0):
            # WARN_A = true;
            if (t1 == 0) and (t2 == 0):
                t1 = 1
                t2 = 1
            elif t1 == 0:
                # return better solution in t1 for m=2 case in invschurc
                t1 = t2
                t2 = 1
            else:  # (t2 == 0)
                t2 = 1
    return [t1, t2]

# #
# # Makes abs(A(1,i)) = abs(A(i,1)) through diagonal similarity for all i.
# #
# function A = deskew(A)
# n = length(A);
# d = ones(n,1);
# for i = 2:n
#     if A(i,1) ~= 0  # don't divide by 0
#         d(i) = sqrt(abs(A(1,i)/A(i,1)));
#         if (d(i) > 1e6)||(d(i) < 1e-6)
#             # something is likely wrong, use 1 instead
#             d(i) = 1;
#         end
#     end     # else leave d(i) = 1
# end
#
# # If D = diag(d), this effectively computes A = D*A*inv(D)
# for i = 2:n
#     A(i,:) = A(i,:)*d(i);
# end
# for i = 2:n
#     A(:,i) = A(:,i)/d(i);
# end
