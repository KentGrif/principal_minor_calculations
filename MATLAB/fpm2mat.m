% FPM2MAT Finds a real or complex matrix that has PM as its 1x1, 2x2,
% and 3x3 principal minors if possible.
%
%   FPM2MAT produces a matrix with some, but perhaps not all, of its
%   principal minors in common with PM.  If the principal minors are
%   not consistent or the matrix that has them is not ODF, FPM2MAT will
%   produce a matrix with different principal minors without warning.
%   Run FMAT2PM on the output matrix A as needed.
%
%   A = FPM2MAT(PM, PMIDX)
%   where PM and PMIDX are in the format produced by FMAT2PM.
%
function [A] = fpm2mat(pm, pmidx)
% Only works on up to 53x53 matrices due to restrictions on indices.

% Since length(pm) = (n^3 + 5*n)/6, the computation below suffices to
% find n given length(pm).
n = floor((6*length(pm))^(1/3));

% initialize globals to allow warnings to be printed only once
global WARN_A WARN_C WARN_I
WARN_A = false; WARN_C = false; WARN_I = false; 

% ipmlevels is a vector of the short indices of the start of each level
% (minus one for convenience of indexing)
ipmlevels = zeros(1,n);
ipmlevels(1) = 0;
for level = 1:n-1;
    ipmlevels(level+1) = ipmlevels(level) + level*(level-1)/2 + 1;
end

% no call to invschurc is necessary for level n-1
nq = n*(n-1)/2 + 1;
q = zeros(1,1,nq);
ipm = ipmlevels(n) + 1;     % short index of current pm
level2 = 2^(n-1);
for i = 1:nq
    if i == 1
        q(1,1,i) = pm(ipm);
    else
        q(1,1,i) = pm(ipm)/pm(pmfind(pmidx, pmidx(ipm) - level2));
    end
    ipm = ipm+1;
end

%
% Main 'level' loop
%
for level = n-2:-1:0        % for consistency with mat2pm levels
    [n1, n1, nq] = size(q);
    nq = nq - level - 1;
    n1 = n1+1;
    qq = zeros(n1, n1, nq);

    ipm = ipmlevels(level+1) + 1;
    level2 = 2^level;
    for i = 1:nq
        if (i == 1)
            pivot = pm(ipm);
        else
            ipm2 = pmfind(pmidx, pmidx(ipm) - level2);
            pivot = pm(ipm)/pm(ipm2);
        end
        iRight = pmfind(pmidx, pmidx(i + ipmlevels(level+2)) + level2);
        if length(iRight) == 1
            iRight = iRight - ipmlevels(level+2);
            qq(:,:,i) = invschurc(pivot, q(:,:,i), q(:,:,iRight));
        else
            qq(:,:,i) = invschurc(pivot, q(:,:,i), ones(n-level-1));
        end
        ipm = ipm+1;
    end
    q = qq;
end
A = q(:,:,1);
A = deskew(A);

if WARN_A
    % ODF (a) not satisfied
    fprintf(2,...
'PM2MAT: off diagonal zeros found, solution suspect.\n');
end
if WARN_C
    fprintf(2,...
'PM2MAT: multiple solutions to make rank(L-R)=1, solution suspect.\n');
end
% disable WARN_I for fast version, since it is routinely triggered

%
% Find i0 in pmidx, returning its index, using a binary search,
% since pmidx is in ascending order.
%
% Same functionality as
%
% find(pmidx == i0)
%
% only faster.
%
function i = pmfind(pmidx, i0)
n = length(pmidx);
iLo = 1;
iHi = n;
if pmidx(iHi) <= i0
    if pmidx(iHi) == i0
        i = n;
    else
        i = [];
    end
    return;
end
iOld = -1;
i = iLo;
while i ~= iOld
    iOld = i;
    i = floor((iHi + iLo)/2);
    if pmidx(i) < i0
        iLo = i;
    elseif pmidx(i) > i0
        iHi = i;
    else
        return;
    end
end
i = [];
return;

% invschurc, solveright and deskew are the same as in pm2mat.m

%
% Suppose A is an (m+1) x (m+1) matrix such that
%
% pivot = A(1,1)
% L = A(2:m+1, 2:m+1)
% R = L - A(2:m+1,1)*A(1,2:m+1)/pivot = (the Schur's complement with
% respect to the pivot or A/A[1]).
%
% Then invschurc finds such an (m+1) x (m+1) matrix A (not necessarily
% unique) given the pivot (a scalar), and the m x m matrices L and R.
%
% If rank(L-R) is not 1, modifies R so that L-R is rank 1.
%
function A = invschurc(pivot, L, R)
global WARN_C WARN_I
myeps_i = 1e-3*norm(R,inf);   % make these relative to magnitude of R
myeps_c = 1e-9*norm(R,inf);
m = length(R);

% Try to make (L-R) rank 1
if m == 2
    [t1,t2] = solveright(L(1,1), L(1,2), L(2,1), L(2,2),...
        R(1,1), R(1,2), R(2,1), R(2,2));

    % This is arbitrary, take the first.
    t = t1;

    R(2,1) = R(2,1)*t;
    R(1,2) = R(1,2)/t;
elseif m >= 3
    % We start with the lower right hand 3x3 submatrix.  We have 3
    % parameters, each with two possible solutions.  Only 1 of the 8
    % possible solutions need give us a L-R which is rank 1.  We find the
    % right solution by brute force.
    i1 = m-2;
    i2 = m-1;
    i3 = m;
    [r1,r2] = solveright(L(i2,i2), L(i2,i3), L(i3,i2), L(i3,i3),...
        R(i2,i2), R(i2,i3), R(i3,i2), R(i3,i3));
    [s1,s2] = solveright(L(i1,i1), L(i1,i2), L(i2,i1), L(i2,i2),...
        R(i1,i1), R(i1,i2), R(i2,i1), R(i2,i2));
    [t1,t2] = solveright(L(i1,i1), L(i1,i3), L(i3,i1), L(i3,i3),...
        R(i1,i1), R(i1,i3), R(i3,i1), R(i3,i3));
    % Perform a parameterized "row reduction" on the first two rows of this
    % matrix and compute the absolute value of the (2,3) entry.  One of
    % them will be nearly zero.
    r111 = abs((L(i2,i3) - R(i2,i3)/r1)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s1)*(L(i1,i3) - R(i1,i3)/t1));
    r112 = abs((L(i2,i3) - R(i2,i3)/r1)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s1)*(L(i1,i3) - R(i1,i3)/t2));
    r121 = abs((L(i2,i3) - R(i2,i3)/r1)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s2)*(L(i1,i3) - R(i1,i3)/t1));
    r122 = abs((L(i2,i3) - R(i2,i3)/r1)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s2)*(L(i1,i3) - R(i1,i3)/t2));
    r211 = abs((L(i2,i3) - R(i2,i3)/r2)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s1)*(L(i1,i3) - R(i1,i3)/t1));
    r212 = abs((L(i2,i3) - R(i2,i3)/r2)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s1)*(L(i1,i3) - R(i1,i3)/t2));
    r221 = abs((L(i2,i3) - R(i2,i3)/r2)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s2)*(L(i1,i3) - R(i1,i3)/t1));
    r222 = abs((L(i2,i3) - R(i2,i3)/r2)*(L(i1,i1) - R(i1,i1)) - ...
        (L(i2,i1) - R(i2,i1)*s2)*(L(i1,i3) - R(i1,i3)/t2));
    rv = [r111, r112, r121, r122, r211, r212, r221, r222];
    mn = min(rv);
    if (r111 == mn)
        r = r1; s = s1; t = t1;
    elseif (r112 == mn)
        r = r1; s = s1; t = t2;
    elseif (r121 == mn)
        r = r1; s = s2; t = t1;
    elseif (r122 == mn)
        r = r1; s = s2; t = t2;
    elseif (r211 == mn)
        r = r2; s = s1; t = t1;
    elseif (r212 == mn)
        r = r2; s = s1; t = t2;
    elseif (r221 == mn)
        r = r2; s = s2; t = t1;
    else % (r222 == mn)
        r = r2; s = s2; t = t2;
    end
    if mn > myeps_i
        WARN_I = true;
    end
    if sum(rv < myeps_c) > 1
        WARN_C = true;
    end
    R(i3,i2) = R(i3,i2)*r;
    R(i2,i3) = R(i2,i3)/r;
    R(i2,i1) = R(i2,i1)*s;
    R(i1,i2) = R(i1,i2)/s;
    R(i3,i1) = R(i3,i1)*t;
    R(i1,i3) = R(i1,i3)/t;

    % Now the lower right hand 3x3 submatrix of L-R has rank 1.  Then we
    % fix up the rest of L-R.
    for i1 = m-3:-1:1
        i2 = i1+1;
        i3 = i1+2;

        % Now the inside lower right submatrix is done, so we
        % only have 2 free parameters and 4 combinations to examine.
        [s1,s2] = solveright(L(i1,i1), L(i1,i2), L(i2,i1), L(i2,i2),...
            R(i1,i1), R(i1,i2), R(i2,i1), R(i2,i2));
        [t1,t2] = solveright(L(i1,i1), L(i1,i3), L(i3,i1), L(i3,i3),...
            R(i1,i1), R(i1,i3), R(i3,i1), R(i3,i3));

        r11 = abs((L(i2,i3) - R(i2,i3))*(L(i1,i1) - R(i1,i1)) - ...
            (L(i2,i1) - R(i2,i1)*s1)*(L(i1,i3) - R(i1,i3)/t1));
        r12 = abs((L(i2,i3) - R(i2,i3))*(L(i1,i1) - R(i1,i1)) - ...
            (L(i2,i1) - R(i2,i1)*s1)*(L(i1,i3) - R(i1,i3)/t2));
        r21 = abs((L(i2,i3) - R(i2,i3))*(L(i1,i1) - R(i1,i1)) - ...
            (L(i2,i1) - R(i2,i1)*s2)*(L(i1,i3) - R(i1,i3)/t1));
        r22 = abs((L(i2,i3) - R(i2,i3))*(L(i1,i1) - R(i1,i1)) - ...
            (L(i2,i1) - R(i2,i1)*s2)*(L(i1,i3) - R(i1,i3)/t2));
        rv = [r11, r12, r21, r22];
        mn = min(rv);
        if (r11 == mn)
            s = s1; t = t1;
        elseif (r12 == mn)
            s = s1; t = t2;
        elseif (r21 == mn)
            s = s2; t = t1;
        else % (r22 == mn)
            s = s2; t = t2;
        end
        if mn > myeps_i
            WARN_I = true;
        end
        if sum(rv < myeps_c) > 1
            WARN_C = true;
        end
        R(i2,i1) = R(i2,i1)*s;
        R(i1,i2) = R(i1,i2)/s;
        R(i3,i1) = R(i3,i1)*t;
        R(i1,i3) = R(i1,i3)/t;
        for j = i1+3:m
            % Finally, once the second row of the submatrix we are working
            % on is uniquely solved, we just pick the solution to the
            % quadratic such that the the first row is a multiple of the
            % second row.  Note that one of r1, r2 will be almost zero.
            % Solving the quadratics leads to much better performance
            % numerically than just taking multiples of the second or
            % any other row.
            %

            j1 = i1+1;
            [t1,t2] = solveright(L(i1,i1), L(i1,j), L(j,i1), L(j,j),...
                R(i1,i1), R(i1,j), R(j,i1), R(j,j));
            r1 = abs((L(j1,j) - R(j1,j))*(L(i1,i1) - R(i1,i1)) - ...
                (L(j1,i1) - R(j1,i1))*(L(i1,j) - R(i1,j)/t1));
            r2 = abs((L(j1,j) - R(j1,j))*(L(i1,i1) - R(i1,i1)) - ...
                (L(j1,i1) - R(j1,i1))*(L(i1,j) - R(i1,j)/t2));
            if (r1 <= r2)
                t = t1;
            else
                t = t2;
            end
            rv = [r1, r2];
            if mn > myeps_i
                WARN_I = true;
            end
            if sum(rv < myeps_c) > 1
                WARN_C = true;
            end

            R(j,i1) = R(j,i1)*t;
            R(i1,j) = R(i1,j)/t;
        end
    end
end

B = (L-R);    % a rank 1 matrix
[mn, idxmax] = max(abs(diag(B)));
% For numerical reasons use the largest diagonal element as a base to find
% the two vectors whose outer product is B*pivot
yT = B(idxmax,:);
if yT(idxmax) == 0
    % This shouldn't happen normally, but to prevent
    % divide by zero when we set all "dependent" principal
    % minors (with index sets greater than or equal to a constant)
    % to the same value, let yT be something.
    yT = ones(1,m);
end
x = B(:,idxmax)*pivot / yT(idxmax);
A = zeros(m+1);
A(1,1) = pivot;
A(1,2:m+1) = yT;
A(2:m+1,1) = x;
A(2:m+1,2:m+1) = L;

%
% Returns the two possible real solutions that will make L-R rank one if we
% let
% r21 = r21*t? (where ? = 1 or 2) and
% r12 = r12/t?
%
function [t1,t2] = solveright(l11,l12,l21,l22,r11,r12,r21,r22)
global WARN_A
x1 = l11-r11;
x2 = l22-r22;
d = sqrt(x1^2*x2^2 + l12^2*l21^2 + r12^2*r21^2 - 2*x1*x2*l12*l21 - ...
    2*x1*x2*r12*r21-2*l12*l21*r21*r12);
if (l12 == 0)||(r21 == 0)
    % This shouldn't happen normally, but to prevent
    % divide by zero when we set all "dependent" principal
    % minors (with index sets greater than or equal to a constant)
    % to the same value, let [t1,t2] be something.
    t1 = 1;
    t2 = 1;
    WARN_A = true;
else
    t1 = (-x1*x2 + l12*l21 + r12*r21 - d)/(2*l12*r21);
    t2 = (-x1*x2 + l12*l21 + r12*r21 + d)/(2*l12*r21);

    % This also shouldn't happen.  Comment above applies.
    if (t1 == 0)||(t2 == 0)
        WARN_A = true;
        if (t1 == 0)&&(t2 == 0)
            t1 = 1;
            t2 = 1;
        elseif (t1 == 0)
            % return better solution in t1 for m=2 case in invschurc
            t1 = t2;
            t2 = 1;
        else  % (t2 == 0)
            t2 = 1;
        end
    end
end

%
% Makes abs(A(1,i)) = abs(A(i,1)) through diagonal similarity for all i.
%
function A = deskew(A)
n = length(A);
d = ones(n,1);
for i = 2:n
    if A(i,1) ~= 0  % don't divide by 0
        d(i) = sqrt(abs(A(1,i)/A(i,1)));
        if (d(i) > 1e6)||(d(i) < 1e-6)
            % something is likely wrong, use 1 instead
            d(i) = 1;
        end
    end     % else leave d(i) = 1
end

% If D = diag(d), this effectively computes A = D*A*inv(D)
for i = 2:n
    A(i,:) = A(i,:)*d(i);
end
for i = 2:n
    A(:,i) = A(:,i)/d(i);
end