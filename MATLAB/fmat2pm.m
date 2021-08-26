% FMAT2PM Finds all 1x1, 2x2 and 3x3 principal minors of an n x n matrix.
%   [PM, PMIDX] = FMAT2PM(A)
%   where "A" is an n x n matrix (zero pivots not handled).
%   MAT2PM returns a vector of all the 1x1, 2x2, and 3x3 principal minors
%   of the matrix "A" in PM.  Also returns PMIDX, a vector of indices
%   that gives the index of the given principal minor in the full
%   binary ordered PM vector that MAT2PM produces.  Thus, for example,
%   if
%
%   A = rand(30);
%   [pm, pmidx] = fmat2pm(A);
%
%   then
%
%   det(A([25 29 30],[25 29 30]))
%
%   is the same as
%
%   pm(find(pmidx == v2idx([25 29 30])))
function [pm, pmidx] = fmat2pm(a)
% Only works on up to 53x53 matrices due to restrictions on indices.
n = length(a);

% nchoosek(n,1) + nchoosek(n,2) + nchoosek(n,3);
pm = zeros(1, (n^2 + 5)*n/6);     % where the principal minors are stored
pmidx = zeros(1, (n^2 + 5)*n/6);  % place to store full (mat2pm) indices
pmidx(1) = 1;
ipm = 1;                    % short (new) index for storing principal minors
q = zeros(n,n,1);           % q is the input queue of unprocessed matrices
q(:,:,1) = a;               % initial queue just has 1 matrix to process

%
% Main 'level' loop
%
for level = 0:n-1
    [n1, n1, nq] = size(q);
    nq2 = nq + level + 1;
    qq = zeros(n1-1, n1-1, nq2);
    ipmlevel = ipm + nq - 1;    % short index of beginning of the level
    ipm2 = 1;
    level2 = 2^level;
    for i = 1:nq
        a = q(:,:,i);
        pm(ipm) = a(1,1);
        ipm1 = pmidx(ipm);      % long index of current pm
        if n1 > 1
            if bitcount(ipm1) < 3
                b = a(2:n1,2:n1);
                % assume all pivots are non-zero
                d = a(2:n1,1)/pm(ipm);
                c = b - d*a(1,2:n1);

                qq(:,:,i) = b;
                pmidx(ipmlevel+i) = ipm1 + level2;
                qq(:,:,nq+ipm2) = c;
                pmidx(ipmlevel+nq+ipm2) = ipm1 + 2*level2;
                ipm2 = ipm2+1;
            else
                b = a(2:n1,2:n1);
                qq(:,:,i) = b;
                pmidx(ipmlevel+i) = ipm1 + level2;
            end
        end

        if i > 1
            pm(ipm) = pm(ipm)*pm(pmfind(pmidx, ipm1 - level2, ipmlevel));
        end
        ipm = ipm + 1;
    end
    q = qq;
end

% Returns the number of bits set in x, or 4 if more than
% 4 are set.
% For example, msb(7) = 3, msb(6) = 2, msb(10) = 2.
function c = bitcount(x)
c = 0;
while x ~= 0
    if bitand(x,1) == 1
        c = c + 1;
        if c >= 4
            return;     % no reason to keep counting
        end
    end
    x = bitshift(x, -1);    % shift right
end

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
function i = pmfind(pmidx, i0, n)
% 1:n is the part of pmidx that has values so far
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