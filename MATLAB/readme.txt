Read me for the MATLAB principal minor computations package
-----------------------------------------------------------

Each of the following *.m files can run independently of all the others.
Place the files with the routines you want to run in any directory in
MATLAB's search path.

`help routine' (from the MATLAB command line) can be used to get the calling
conventions for each routine.

Contents
--------
mat2pm      Computes all the principal minors of an input matrix.
pm2mat      Finds a matrix that has the input set of principal minors as 
            its principal minors, if possible.
pmfront     Front end that calls pm2mat and checks to see if the resulting
            matrix realizes the input set of principal minors.
fmat2pm     Computes the 1x1, 2x2 and 3x3 principal minors of an input 
            matrix.
fpm2mat     Finds a matrix that has the input set of 1x1, 2x2 and 3x3 
            principal minors as its principal minors, if possible.
getpm       Gets a given principal minor from a vector of principal minors
            produced by mat2pm given a vector representing an index set.
idx2v       Given an index into a vector of principal minors, returns the 
            index set of the principal minor it represents.
v2idx       Given an index set of a principal minor, produces its index in
            the output vector of mat2pm.
pmshow      Displays the output of mat2pm with its index number and index 
            sets.
