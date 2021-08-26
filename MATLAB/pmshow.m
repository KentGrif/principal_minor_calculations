% PMSHOW  Displays the given set of principal minors with its index number
% and index sets.
function pmshow(pm)
for i = 1:length(pm)
    v = idx2v(i);
    if imag(pm(i)) == 0
        fprintf(1,'%d\t[%14s]\t%g\n', i, int2str(v), pm(i));
    else % display complex principal minor
        if imag(pm(i)) > 0
            fprintf(1,'%d\t[%14s]\t%g + %gi\n', i, int2str(v),...
                real(pm(i)), imag(pm(i)));
        else
            fprintf(1,'%d\t[%14s]\t%g - %gi\n', i, int2str(v),...
                real(pm(i)), -imag(pm(i)));
        end
    end
end

% IDX2V  Converts a MAT2PM index into a set of principal minors pm to
% an index set that corresponds to the given principal minor.
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
