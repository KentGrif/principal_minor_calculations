% PMFRONT  Finds a real or complex matrix that has PM as its principal
% minors if possible, prints out a warning if no such matrix can
% be found by PM2MAT.
%
%   A = PMFRONT(PM)
%   where PM is a 2^n - 1 vector of principal minors and A will be an n x n
%   matrix.
%
%   The structure of PM, where |A[v]| is the principal minor of "A" indexed
%   by the vector v:
%   PM: |A[1]| |A[2]| |A[1 2]| |A[3]| |A[1 3]| |A[2 3]| |A[1 2 3]| ...
function A = pmfront(pm)
myeps = 1e-5;   % tolerance for relative errors in the principal minors

% First run pm2mat
A = pm2mat(pm);

% Next, run mat2pm on the result
pm1 = mat2pm(A);

smallestpm = min(abs(pm));
if smallestpm < 1e-10
    fprintf(2, ...
'There are principal minors very close to zero, relative errors in\n');
    fprintf(2, ...
'principal minors may not be meaningful.  Consider the absolute error\n')
    fprintf(2, ...
'to decide if PM2MAT succeeded.\n');
    err = norm((pm-pm1),inf);
    fprintf(2, ...
'The maximum absolute error in the principal minors is %e\n', err);
else
    % Compare the results in terms of the relative error in the pm's
    err = norm((pm-pm1)./abs(pm),inf);
    if err > myeps
        fprintf(2, 'PM2MAT failed\n');
        fprintf(2, ...
'No matrix could be found that has all the requested principal minors.\n');
        fprintf(2, ...
'The PM''s are inconsistent or they come from a non-ODF matrix.\n');
    else
        fprintf(2, 'PM2MAT succeeded\n');
    end
    fprintf(2, ...
'The maximum relative error in the principal minors is %e\n', err);
end
