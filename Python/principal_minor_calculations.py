import numpy as np


# Default data type for computation
DTYPE = np.cdouble     # complex number, represented by two double-precision floats


def mat2pm(a: np.array, thresh=1e-5):
    """
    mat2pm returns a 2^n - 1 vector of all the principal minors of the matrix a.
    :param a: n x n input matrix
    :param thresh: Threshold for psuedo-pivoting.  Pseudo-pivoting will occur when a pivot smaller in magnitude than
        thresh arises.  Set thresh = 0 to never pseudo-pivot except for a pivot of exactly zero.
    :return: Tuple of (array of principal minors: np.array, computation info: str).
        The structure of pm, where |a[v]| is the principal minor of "a" indexed by the vector v is:
        pm: |a[1]|, |a[2]|, |a[1 2]|, |a[3]|, |a[1 3]|, |a[2 3]|, |a[1 2 3]|,...
    """
    assert(len(a.shape) == 2 and a.shape[0] == a.shape[1])
    a = a.astype(DTYPE)
    n = a.shape[0]
    scale = np.sum(abs(a))/(n*n)    # average magnitude of matrix
    if scale == 0:
        scale = 1               # prevent divide by 0 if matrix is zero
    ppivot = scale              # value to use as a pivot if near 0 pivot arises

    zeropivs = []
    pm = np.zeros(2**n - 1, dtype=DTYPE)     # where the principal minors are stored
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
                b = a[1:, 1:]
                d = a[1:, 0] / pm[ipm]
                c = b - np.outer(d, a[0, 1:])

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
        for j in range(mask+delta2-1, len(pm), delta2):
            pm[j] -= ppivot*pm[j-delta]

    return pm, f'mat2pm: pseudo-pivoted {len(zeropivs)} times, smallest pivot used: {pivmin}'


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


def pm2mat(pm: np.array):
    """
    pm2mat Finds a real or complex matrix that has pm as its principal
    minors.
    pm2mat produces a matrix with some, but perhaps not all, of its
    principal minors in common with pm.  If the principal minors are
    not consistent or the matrix that has them is not ODF, pm2mat will
    produce a matrix with different principal minors without throwing exceptions.
    Run mat2pm on the output matrix a as needed.
    :param pm: The structure of pm, where |a[v]| is the principal minor of "a" indexed by the vector v is:
        pm: |a[1]|, |a[2]|, |a[1 2]|, |a[3]|, |a[1 3]|, |a[2 3]|, |a[1 2 3]|,...
    :return: Tuple of (a: np.array, computation info: str)
    """
    warn_not_odf = False
    warn_under_determined = False
    warn_inconsistent = False

    def _invschurc(_pivot, _l, _r):
        """
        Suppose a is an (m+1) x (m+1) matrix such that

        pivot = a[0, 0]
        L = a[1:m+1, 1:m+1]
        R = L - a[1:m+1, 0]*a[0, 1:m+1)/pivot = (the Schur's complement with
        respect to the pivot or A/A[0]).

        Then invschurc finds such an (m+1) x (m+1) matrix a (not necessarily
        unique) given the pivot (a scalar), and the m x m matrices L and R.

        If rank(L-R) is not 1, modifies R so that L-R is rank 1.

        :param _pivot: (0, 0) element of matrix a
        :param _l: a[1:m+1, 1:m+1]
        :param _r: _l - a[1:m+1, 0]*a[0, 1:m+1)/_pivot
        :return: Inverse Schur's complement
        """
        nonlocal warn_under_determined
        nonlocal warn_inconsistent

        # max allowable ratio of nearly zero row value to next largest row value
        myeps_i = 1e-3

        # make this relative to magnitude of R
        myeps_c = 1e-12 * np.linalg.norm(_r, ord=np.inf)
        m = _r.shape[0]

        # Try to make (L-R) rank 1
        if m == 2:
            [t1, t2] = _solveright(_l[0, 0], _l[0, 1], _l[1, 0], _l[1, 1], _r[0, 0], _r[0, 1], _r[1, 0], _r[1, 1])

            # This is arbitrary, take the first.
            t = t1

            _r[1, 0] = _r[1, 0] * t
            _r[0, 1] = _r[0, 1] / t
        elif m >= 3:
            # We start with the lower right hand 3x3 submatrix.  We have 3
            # parameters, each with two possible solutions.  Only 1 of the 8
            # possible solutions need give us a L-R which is rank 1.  We find the
            # right solution by brute force.
            i1 = m - 3
            i2 = m - 2
            i3 = m - 1
            [r1, r2] = _solveright(_l[i2, i2], _l[i2, i3], _l[i3, i2], _l[i3, i3], _r[i2, i2], _r[i2, i3], _r[i3, i2],
                                   _r[i3, i3])
            [s1, s2] = _solveright(_l[i1, i1], _l[i1, i2], _l[i2, i1], _l[i2, i2], _r[i1, i1], _r[i1, i2], _r[i2, i1],
                                   _r[i2, i2])
            [t1, t2] = _solveright(_l[i1, i1], _l[i1, i3], _l[i3, i1], _l[i3, i3], _r[i1, i1], _r[i1, i3], _r[i3, i1],
                                   _r[i3, i3])
            # Perform a parameterized "row reduction" on the first two rows of this
            # matrix and compute the absolute value of the (2,3) entry.  One of
            # them will be nearly zero.
            r111 = abs((_l[i2, i3] - _r[i2, i3] / r1) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s1) * (_l[i1, i3] - _r[i1, i3] / t1))
            r112 = abs((_l[i2, i3] - _r[i2, i3] / r1) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s1) * (_l[i1, i3] - _r[i1, i3] / t2))
            r121 = abs((_l[i2, i3] - _r[i2, i3] / r1) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s2) * (_l[i1, i3] - _r[i1, i3] / t1))
            r122 = abs((_l[i2, i3] - _r[i2, i3] / r1) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s2) * (_l[i1, i3] - _r[i1, i3] / t2))
            r211 = abs((_l[i2, i3] - _r[i2, i3] / r2) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s1) * (_l[i1, i3] - _r[i1, i3] / t1))
            r212 = abs((_l[i2, i3] - _r[i2, i3] / r2) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s1) * (_l[i1, i3] - _r[i1, i3] / t2))
            r221 = abs((_l[i2, i3] - _r[i2, i3] / r2) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s2) * (_l[i1, i3] - _r[i1, i3] / t1))
            r222 = abs((_l[i2, i3] - _r[i2, i3] / r2) * (_l[i1, i1] - _r[i1, i1]) -
                       (_l[i2, i1] - _r[i2, i1] * s2) * (_l[i1, i3] - _r[i1, i3] / t2))
            rv = sorted([r111, r112, r121, r122, r211, r212, r221, r222])
            mn = rv[0]
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

            if mn > rv[1] * myeps_i:
                warn_inconsistent = True

            if np.count_nonzero(rv < myeps_c) > 1:
                warn_under_determined = True

            _r[i3, i2] = _r[i3, i2] * r
            _r[i2, i3] = _r[i2, i3] / r
            _r[i2, i1] = _r[i2, i1] * s
            _r[i1, i2] = _r[i1, i2] / s
            _r[i3, i1] = _r[i3, i1] * t
            _r[i1, i3] = _r[i1, i3] / t

            # Now the lower right hand 3x3 submatrix of L-R has rank 1.  Then we
            # fix up the rest of L-R.
            for i1 in range(m - 4, -1, -1):
                i2 = i1 + 1
                i3 = i1 + 2

                # Now the inside lower right submatrix is done, so we
                # only have 2 free parameters and 4 combinations to examine.
                [s1, s2] = _solveright(_l[i1, i1], _l[i1, i2], _l[i2, i1], _l[i2, i2], _r[i1, i1], _r[i1, i2],
                                       _r[i2, i1], _r[i2, i2])
                [t1, t2] = _solveright(_l[i1, i1], _l[i1, i3], _l[i3, i1], _l[i3, i3], _r[i1, i1], _r[i1, i3],
                                       _r[i3, i1], _r[i3, i3])

                r11 = abs((_l[i2, i3] - _r[i2, i3]) * (_l[i1, i1] - _r[i1, i1]) -
                          (_l[i2, i1] - _r[i2, i1] * s1) * (_l[i1, i3] - _r[i1, i3] / t1))
                r12 = abs((_l[i2, i3] - _r[i2, i3]) * (_l[i1, i1] - _r[i1, i1]) -
                          (_l[i2, i1] - _r[i2, i1] * s1) * (_l[i1, i3] - _r[i1, i3] / t2))
                r21 = abs((_l[i2, i3] - _r[i2, i3]) * (_l[i1, i1] - _r[i1, i1]) -
                          (_l[i2, i1] - _r[i2, i1] * s2) * (_l[i1, i3] - _r[i1, i3] / t1))
                r22 = abs((_l[i2, i3] - _r[i2, i3]) * (_l[i1, i1] - _r[i1, i1]) -
                          (_l[i2, i1] - _r[i2, i1] * s2) * (_l[i1, i3] - _r[i1, i3] / t2))
                rv = sorted([r11, r12, r21, r22])
                mn = rv[0]
                if r11 == mn:
                    [s, t] = [s1, t1]
                elif r12 == mn:
                    [s, t] = [s1, t2]
                elif r21 == mn:
                    [s, t] = [s2, t1]
                else:  # (r22 == mn)
                    [s, t] = [s2, t2]

                if mn > rv[1] * myeps_i:
                    warn_inconsistent = True

                if np.count_nonzero(rv < myeps_c) > 1:
                    warn_under_determined = True

                _r[i2, i1] = _r[i2, i1] * s
                _r[i1, i2] = _r[i1, i2] / s
                _r[i3, i1] = _r[i3, i1] * t
                _r[i1, i3] = _r[i1, i3] / t
                for _j in range(i1 + 2, m):
                    # Finally, once the second row of the submatrix we are working
                    # on is uniquely solved, we just pick the solution to the
                    # quadratic such that the the first row is a multiple of the
                    # second row.  Note that one of r1, r2 will be almost zero.
                    # Solving the quadratics leads to much better performance
                    # numerically than just taking multiples of the second or
                    # any other row.
                    #

                    j1 = i1 + 1
                    [t1, t2] = _solveright(_l[i1, i1], _l[i1, _j], _l[_j, i1], _l[_j, _j], _r[i1, i1], _r[i1, _j],
                                           _r[_j, i1], _r[_j, _j])
                    r1 = abs((_l[j1, _j] - _r[j1, _j]) * (_l[i1, i1] - _r[i1, i1]) -
                             (_l[j1, i1] - _r[j1, i1]) * (_l[i1, _j] - _r[i1, _j] / t1))
                    r2 = abs((_l[j1, _j] - _r[j1, _j]) * (_l[i1, i1] - _r[i1, i1]) -
                             (_l[j1, i1] - _r[j1, i1]) * (_l[i1, _j] - _r[i1, _j] / t2))
                    if r1 <= r2:
                        t = t1
                    else:
                        t = t2

                    rv = sorted([r1, r2])
                    mn = rv[0]
                    if mn > rv[1] * myeps_i:
                        warn_inconsistent = True

                    if np.count_nonzero(rv < myeps_c) > 1:
                        warn_under_determined = True

                    _r[_j, i1] = _r[_j, i1] * t
                    _r[i1, _j] = _r[i1, _j] / t

        b = (_l - _r)  # B is a rank 1 matrix
        idxmax = np.argmax(abs(np.diagonal(b)))
        # For numerical reasons use the largest diagonal element as a base to find
        # the two vectors whose outer product is B*pivot
        y_t = b[idxmax, :]
        if y_t[idxmax] == 0:
            # This shouldn't happen normally, but to prevent
            # divide by zero when we set all "dependent" principal
            # minors (with index sets greater than or equal to a constant)
            # to the same value, let yT be something.
            y_t = np.ones(m)

        x = b[:, idxmax] * _pivot / y_t[idxmax]
        _a = np.zeros((m + 1, m + 1), dtype=DTYPE)
        _a[0, 0] = _pivot
        _a[0, 1:m + 1] = y_t
        _a[1:m + 1, 0] = x
        _a[1:m + 1, 1:m + 1] = _l
        return _a

    def _solveright(_l11, _l12, _l21, _l22, _r11, _r12, _r21, _r22):
        """
        Returns the two possible real solutions that will make L-R rank one if we let
        r21 = r21*t? (where ? = 1 or 2) and
        r12 = r12/t?

        :param _l11:
        :param _l12:
        :param _l21:
        :param _l22:
        :param _r11:
        :param _r12:
        :param _r21:
        :param _r22:
        :return: Tuple of real solutions
        """
        nonlocal warn_not_odf
        _x1 = _l11 - _r11
        _x2 = _l22 - _r22
        _d = np.sqrt(_x1*_x1*_x2*_x2 + _l12*_l12*_l21*_l21 + _r12*_r12*_r21*_r21 - 2*_x1*_x2*_l12*_l21 -
                     2*_x1*_x2*_r12*_r21 - 2*_l12*_l21*_r21*_r12)
        if (_l12 == 0) or (_r21 == 0):
            # This shouldn't happen normally, but to prevent
            # divide by zero when we set all "dependent" principal
            # minors (with index sets greater than or equal to a constant)
            # to the same value, let [t1,t2] be something.
            _t1 = 1
            _t2 = 1
            warn_not_odf = True
        else:
            _t1 = (-_x1*_x2 + _l12*_l21 + _r12*_r21 - _d) / (2*_l12*_r21)
            _t2 = (-_x1*_x2 + _l12*_l21 + _r12*_r21 + _d) / (2*_l12*_r21)

            # This also shouldn't happen.  Comment above applies.
            if _t1 == 0 or _t2 == 0:
                warn_not_odf = True
                if _t1 == 0 and _t2 == 0:
                    _t1 = 1
                    _t2 = 1
                elif _t1 == 0:
                    # return better solution in t1 for m=2 case in invschurc
                    _t1 = _t2
                    _t2 = 1
                else:  # t2 == 0
                    _t2 = 1
        return _t1, _t2

    myeps = 1e-10
    pm = pm.astype(DTYPE)
    n = int(round(np.log2(len(pm)+1)))

    # Make first (smallest) entry of zeropivs an impossible index
    zeropivs = [-1]
    # Pick a random pseudo-pivot value that minimizes the chances that
    # pseudo-pivoting will create a non-ODF matrix.
    ppivot = 1.9501292851471754

    # To avoid division by zero, do an operation analogous to the zeropivot
    # loop in mat2pm.
    _pm = np.copy(pm)
    for i in range((len(_pm)+1)//2 - 1):
        if np.abs(_pm[i]) < myeps:
            mask = i + 1
            zeropivs.append(i)
            ipm1 = mask & ~_msb(mask)
            if ipm1 == 0:
                _pm[i] += ppivot
            else:
                _pm[i] = (_pm[i]/_pm[ipm1-1] + ppivot)*_pm[ipm1-1]
            delta = _msb(mask)
            delta2 = 2*delta
            for j in range(mask+delta2-1, len(_pm), delta2):
                _pm[j] += ppivot*_pm[j - delta]
    zeropivsidx = len(zeropivs) - 1
    zeropivsmax = zeropivs[zeropivsidx]
    zeropivsidx -= 1

    # initial processing is special, no call to invschurc is necessary
    nq = (len(_pm)+1) // 2
    n1 = 1
    q = [None] * nq   # zeros(1,1,nq)
    ipm1 = nq-1
    ipm2 = 0
    for i in range(nq):
        if i == 0:
            pivot = _pm[ipm1]
            q[i] = np.array([[pivot]], dtype=DTYPE)
        else:
            pivot = _pm[ipm1]/_pm[ipm2]
            q[i] = np.array([[pivot]], dtype=DTYPE)
            ipm2 += 1
        ipm1 += 1

    #
    # Main 'level' loop
    #
    pivmin = np.finfo(np.float64).max
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
                pivot = _pm[ipm1]
            else:
                pivot = _pm[ipm1]/_pm[ipm2]
                ipm2 -= 1
            pivmin = min(pivmin, np.abs(pivot))
            qq[i] = _invschurc(pivot, q[i], q[i+nq])
            if zeropivsmax == ipm1:
                qq[i][0][0] -= ppivot
                zeropivsmax = zeropivs[zeropivsidx]
                zeropivsidx -= 1
            ipm1 -= 1
        q = qq
    a = q[0]
    _deskew(a)
    warn = f'pm2mat: pseudo-pivoted {len(zeropivs)-1} times, smallest pivot used: {pivmin}'
    if warn_not_odf:
        warn = (warn + ';  ' if warn else '') + 'pm2mat: off diagonal zeros found, solution suspect.'
    if warn_under_determined:
        warn = (warn + ';  ' if warn else '') + 'pm2mat: multiple solutions to make rank(L-R)=1, solution suspect.'
    if warn_inconsistent:
        warn = (warn + ';  ' if warn else '') + 'pm2mat: input principal minors may be inconsistent, solution suspect.'
    return a, warn


#
#
#
def _deskew(_a: np.array):
    """
    Makes abs(a[0, i]) = abs(a[i, 0]) through diagonal similarity for all i.
    :param _a: Square matrix
    :return: Deskewed square matrix (but also modifies _a by reference)
    """
    assert(len(_a.shape) == 2 and _a.shape[0] == _a.shape[1])
    _n = _a.shape[0]
    _d = np.ones(_n)
    for _i in range(1, _n):
        if _a[_i, 0] != 0:  # don't divide by 0
            _d[_i] = np.sqrt(np.abs(_a[0, _i]/_a[_i, 0]))
            if _d[_i] > 1e6 or _d[_i] < 1e-6:
                # something is likely wrong, use 1 instead
                _d[_i] = 1.
        # else leave d(i) = 1

    # If D = diag(d), this effectively computes A = D*A*inv(D)
    for _i in range(1, _n):
        _a[_i, :] = _a[_i, :]*_d[_i]
    for _i in range(1, _n):
        _a[:, _i] = _a[:, _i]/_d[_i]
    return _a
