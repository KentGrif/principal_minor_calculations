# principal_minor_calculations
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About Principal Minor Calculations</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
<br>


## About Principal Minor Calculations
This project contains algorithms to compute all the principal minors of a square matrix, and, under certain circumstances, solve the inverse problem of finding a square matrix (not unique) that has a given set of principal minors.  The research that formed the basis of these computations was performed at Washington State University under the guidance of [Dr. Michael Tsatsomeros](http://www.math.wsu.edu/faculty/tsat/).  It was written up in two papers in *Linear Algebra and Its Applications*:

- [https://doi.org/10.1016/j.laa.2006.04.008](https://doi.org/10.1016/j.laa.2006.04.008)
- [https://doi.org/10.1016/j.laa.2006.04.009](https://doi.org/10.1016/j.laa.2006.04.009)

which are also available on the websites of the authors:

- [http://www.math.wsu.edu/faculty/tsat/pm.html](http://www.math.wsu.edu/faculty/tsat/pm.html)
- [https://www.kentegriffin.com/publications](https://www.kentegriffin.com/publications)

<p align="right">(<a href="#top">back to top</a>)</p>


### Built With

The original work was all done in MATLAB.  More recently, I've added a port of the two primary algorithms, MAT2PM and PM2MAT, in Python.

* [MATLAB](https://www.mathworks.com/products/matlab.html)
* [Python](https://www.python.org/)

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

In MATLAB, after downloading the *.m files and placing them in your MATLAB search path, you can define a square matrix, compute its principal minors with mat2pm (in the order described in the papers, the final principal minor is the determinant of the matrix), and then find a matrix that has those principal minors with pm2mat:

<pre>
>> m = [[1, 2, 6]; [2, 4, 5]; [-1, 2, 3]]
m =

   1   2   6
   2   4   5
  -1   2   3

>> pm = mat2pm(m)
MAT2PM: pseudo-pivoted 1 times, smallest pivot used: 1.000000e+00
pm =

    1    4    0    3    9    2   28

>> pm2mat(pm)
ans =

   1.0000   2.0000  -2.4495
   2.0000   4.0000   4.8990
   2.4495   2.0412   3.0000
</pre>
<br>

In Python, after downloading and importing principal_minor_calculations.py, you can define a square numpy array, compute its principal minors with mat2pm, and then find a matrix (numpy array) that has those principal minors with pm2mat:

<pre>
In [1]: import numpy as np

In [2]: import principal_minor_calculations as pmc

In [3]: m = np.array([[1, 2, 6], [2, 4, 5], [-1, 2, 3]])

In [4]: m
Out[4]: 
array([[ 1,  2,  6],
       [ 2,  4,  5],
       [-1,  2,  3]])

In [5]: pm, info = pmc.mat2pm(m)

In [6]: pm
Out[6]: array([ 1.+0.j,  4.+0.j,  0.+0.j,  3.+0.j,  9.+0.j,  2.+0.j, 28.+0.j])

In [7]: info
Out[7]: 'mat2pm: pseudo-pivoted 1 times, smallest pivot used: 1.0'

In [8]: m2, info = pmc.pm2mat(pm)

In [9]: m2
Out[9]: 
array([[ 1.        +0.j,  2.        +0.j, -2.44948974+0.j],
       [ 2.        +0.j,  4.        +0.j,  4.89897949+0.j],
       [ 2.44948974+0.j,  2.04124145+0.j,  3.        +0.j]])

In [10]: m2.astype(float)
/home/kent/anaconda3/bin/ipython:1: ComplexWarning: Casting complex values to real discards the imaginary part
Out[10]: 
array([[ 1.        ,  2.        , -2.44948974],
       [ 2.        ,  4.        ,  4.89897949],
       [ 2.44948974,  2.04124145,  3.        ]])
</pre>

As explained in the papers, the natural field for this computation is the field of complex numbers, but the results can be converted to floats with *astype* if the imaginary parts are negligible as shown above.

<p align="right">(<a href="#top">back to top</a>)</p>


## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


## Contact

Kent Griffin - [website](https://www.kentegriffin.com/) - kentgrif (at) gmail (dot) com<br>
Michael Tsatsomeros - [website](http://www.math.wsu.edu/faculty/tsat/) -  tsat (at) wsu (dot) edu<br>

Project Link: [https://github.com/KentGrif/principal_minor_calculations](https://github.com/KentGrif/principal_minor_calculations)

<p align="right">(<a href="#top">back to top</a>)</p>


## Acknowledgments

Thanks to Andrew James for Python review and packaging.
<p align="right">(<a href="#top">back to top</a>)</p>
