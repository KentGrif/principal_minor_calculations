% GETPM  Extracts a desired principal minor from the PM structure produced
% by MAT2PM.
%   PMINOR = GETPM(PM, V)
%   where PM is a vector of 2^n - 1 principal minors in binary order
%   produced by MAT2PM and V is the vector of the index set.  The elements
%   of V need not be sorted, but they must all be unique.
%   
%   Example: If
%   A = rand(6);
%   pm = mat2pm(A);
%
%   then
%
%   getpm(pm, [1, 3, 5])
%
%   produces the same result as
%
%   det(A([1, 3, 5],[1, 3, 5]))
function [pminor] = getpm(pm, v)
% The index into pm is simply the binary number with the v(i)'th bit set
% for each i.
n = length(v);          % length of vector containing indices of minor
idx = 0;
for i = 1:n
    idx = idx + bitshift(1,v(i)-1);
end
pminor = pm(idx);