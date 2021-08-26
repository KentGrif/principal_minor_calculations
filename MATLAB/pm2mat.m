% PM2MAT Finds a real or complex matrix that has PM as its principal
% minors.
%
%   PM2MAT produces a matrix with some, but perhaps not all, of its
%   principal minors in common with PM.  If the principal minors are
%   not consistent or the matrix that has them is not ODF, PM2MAT will
%   produce a matrix with different principal minors without warning.
%   Run MAT2PM on the output matrix A as needed.
%
%   A = PM2MAT(PM)
%   where PM is a 2^n - 1 vector of principal minors and A will be an n x n
%   matrix.
%
%   The structure of PM, where |A[v]| is the principal minor of "A" indexed
%   by the vector v:
%   PM: |A[1]| |A[2]| |A[1 2]| |A[3]| |A[1 3]| |A[2 3]| |A[1 2 3]| ...
function A = pm2mat(pm)
myeps = 1e-10;

n = log2(length(pm)+1);

% Make first (smallest) entry of zeropivs an impossible index
zeropivs = 0;
% Pick a random pseudo-pivot value that minimizes the chances that
% pseudo-pivoting will create a non-ODF matrix.
ppivot = 1.9501292851471754e+000;

% initialize globals to allow warnings to be printed only once
global WARN_A WARN_C WARN_I
WARN_A = false; WARN_C = false; WARN_I = false;

% To avoid division by zero, do an operation analogous to the zeropivot
% loop in mat2pm.
for i = 1:((length(pm)+1)/2 - 1)
    if (abs(pm(i)) < myeps)
        mask = i;
        zeropivs = union(zeropivs, i);
        ipm1 = bitand(mask, bitcmp(msb(mask),48));
        if ipm1 == 0
            pm(mask) = pm(mask) + ppivot;
        else
            pm(mask) = (pm(mask)/pm(ipm1) + ppivot)*pm(ipm1);
        end
        delta = msb(mask);
        delta2 = 2*delta;
        for j = mask+delta2:delta2:2^n - 1
            pm(j) = pm(j) + ppivot*pm(j - delta);
        end
    end
end
zeropivsidx = length(zeropivs) - 1;
zeropivsmax = zeropivs(zeropivsidx+1);

% initial processing is special, no call to invschurc is necessary
nq = 2^(n-1);
q = zeros(1,1,nq);
ipm1 = nq;
ipm2 = 1;
for i = 1:nq
    if i == 1
        q(1,1,i) = pm(ipm1);
    else
        q(1,1,i) = pm(ipm1)/pm(ipm2);
        ipm2 = ipm2+1;
    end
    ipm1 = ipm1+1;
end

%
% Main 'level' loop
%
for level = n-2:-1:0        % for consistency with mat2pm levels
    [n1, n1, nq] = size(q);
    nq = nq/2;
    n1 = n1+1;
    % The output queue has half the number of matrices, each one larger in
    % row and col dimension
    qq = zeros(n1, n1, nq);

    ipm1 = 2*nq-1;
    ipm2 = nq-1;
    for i = nq:-1:1     % process matrices in reverse order for zeropivs
        if (i == 1)
            pivot = pm(ipm1);
        else
            pivot = pm(ipm1)/pm(ipm2);
            ipm2 = ipm2-1;
        end
        qq(:,:,i) = invschurc(pivot, q(:,:,i), q(:,:,i+nq));
        if zeropivsmax == ipm1
            qq(1,1,i) = qq(1,1,i) - ppivot;
            zeropivsmax = zeropivs(zeropivsidx);
            zeropivsidx = zeropivsidx - 1;
        end
        ipm1 = ipm1-1;
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
if WARN_I
    fprintf(2, ...
'PM2MAT: input principal minors may be inconsistent, solution suspect.\n');
end

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

% Returns the numerical value of the most significant bit of x.
% For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.
function m = msb(x)
persistent MSBTABLE     % MSBTABLE persists between calls to mat2pm
if isempty(MSBTABLE)
    % If table is empty, initialize it
    MSBTABLE = zeros(255,1);
    for i=1:255
        MSBTABLE(i) = msbslow(i);
    end
end

m = 0;
% process 8 bits at a time for speed
if x ~= 0
    while x ~= 0
        x1 = x;
        x = bitshift(x, -8);    % 8 bit left shift
        m = m + 8;
    end
    m = bitshift(MSBTABLE(x1), m-8); % right shift
end

% Returns the numerical value of the most significant bit of x.
% For example, msb(7) = 4, msb(6) = 4, msb(13) = 8.  Slow version
% used to build a table.
function m = msbslow(x)
m = 0;
if x ~= 0
    m = 1;
    while  x ~= 0
        x = bitshift(x, -1);
        m = 2*m;
    end
    m = m/2;
end
