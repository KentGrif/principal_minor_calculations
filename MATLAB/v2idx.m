% V2IDX  Converts a MAT2PM index set (vector) to the index in pm that
% corresponds to a given principal minor.
% 
% For example, if
%
% A = rand(4)
% pm = mat2pm(A)
% idx = v2idx([1 3 4])
%
% then idx = 13 and
% 
% det(A([1 3 4],[1 3 4]))
% 
% will equal
% 
% pm(idx)
%
function idx = v2idx(v)
% The index into pm is simply the binary number with the v(i)'th bit set
% for each i.
n = length(v);          % length of vector containing indices of minor
idx = 0;
for i = 1:n
    idx = idx + bitshift(1,v(i)-1);
end