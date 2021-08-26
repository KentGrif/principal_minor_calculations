% IDX2V  Converts a MAT2PM index into a set of principal minors PM to
% an index set that corresponds to the given principal minor.
% 
% For example, if
%
% A = rand(4)
% pm = mat2pm(A)
% v = idx2v(13)
%
% then v = [1 3 4] and
% 
% det(A(v,v))
% 
% will equal
% 
% pm(13)
%
function v = idx2v(idx)
v = [];
i = 1;
while idx ~= 0
    if bitand(idx, 1) ~= 0
        v = [v i];
    end
    idx = bitshift(idx, -1);    % shift by 1 to the right
    i = i+1;
end
