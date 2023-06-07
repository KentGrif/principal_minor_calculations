from cProfile import Profile
from pstats import Stats

import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from principal_minor_calculations import mat2pm, pm2mat, _msb, PMInfo, pm_info_to_string


def test_msb():
    assert_equal(_msb(1), 1)
    assert_equal(_msb(2), 2)
    assert_equal(_msb(7), 4)
    assert_equal(_msb(6), 4)
    assert_equal(_msb(13), 8)
    assert_equal(_msb(0x1234567812345678), 0x1000000000000000)
    assert_equal(_msb(0xf234567812345678), 0x8000000000000000)


def test_pm_calcs():
    m1 = np.array([[1, 2],
                   [3, 4]])
    pm1, info = mat2pm(m1)
    assert_array_equal(pm1, [1, 4, -2])
    m2, info = pm2mat(pm1)
    pm2, info = mat2pm(m2)
    assert_array_equal(pm1, pm2)

    m1 = np.array([[1, 2, 6],
                   [2, 4, 5],
                   [-1, 2, 3]])
    pm1, info = mat2pm(m1)
    assert_array_equal(pm1, [1, 4, 0, 3, 9, 2, 28])
    m2, info = pm2mat(pm1)
    pm2, info = mat2pm(m2)
    assert_array_almost_equal(pm1, pm2, decimal=10)

    m1 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0]])
    pm1, info = mat2pm(m1)
    assert_array_equal(pm1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    assert_equal(info.number_of_times_ppivoted, 7)
    assert_equal(info.smallest_pivot, 0.25)
    m2, info = pm2mat(pm1)
    assert_equal(info.number_of_times_ppivoted, 0)
    assert_almost_equal(info.smallest_pivot, 1.9501292851471754, decimal=10)
    assert_array_equal(m2, [[0, 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0]])
    assert info.warn_not_odf
    assert info.warn_under_determined
    assert not info.warn_inconsistent

    pm_inconsistent = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    m, info = pm2mat(pm_inconsistent)
    assert not info.warn_not_odf
    assert not info.warn_under_determined
    assert info.warn_inconsistent
    pm, info = mat2pm(m)
    # all but two of the principal minors actually match
    pm[12] = 6.
    pm[14] = 8.
    assert_array_almost_equal(pm, pm_inconsistent, decimal=10)

    m1 = np.array([[-6, 3, -9, 4],
                   [-6, -5, 3, 6],
                   [3, -3, 6, -7],
                   [1, 1, -1, -3]])
    pm1, info = mat2pm(m1)
    pm_truth = np.array([-6, -5, 48, 6, -9, -21, -36, -3, 14, 9, -94, -25, 96, 59, 6])
    assert_equal(info.number_of_times_ppivoted, 0)
    assert_equal(info.smallest_pivot, 0.75)
    assert_array_almost_equal(pm1, pm_truth, decimal=10)
    m2, info = pm2mat(pm_truth)
    pm2, info = mat2pm(m2)
    assert_array_almost_equal(pm1, pm2, decimal=10)

    m1 = np.array([[1 + 1j,  3 - 4j, -2 - 1j],
                   [7 + 3j, -5 + 2j, -1 - 1j],
                   [0 - 4j,  3 + 0j,  3 + 3j]])
    pm1, info = mat2pm(m1)
    pm_truth = np.array([1.0 + 1.0j, -5.0 + 2.0j, -40.0 + 16.0j, 3.0 + 3.0j, 4.0 - 2.0j, -18.0 - 6.0j, -201.0 - 29.0j])
    assert_equal(info.number_of_times_ppivoted, 0)
    assert_almost_equal(info.smallest_pivot, np.sqrt(2))
    assert_array_almost_equal(pm1, pm_truth, decimal=10)
    m2, info = pm2mat(pm_truth)
    pm2, info = mat2pm(m2)
    assert_array_almost_equal(pm1, pm2, decimal=10)

    n = 18
    m1 = np.random.rand(n, n)

    do_profile = False
    if do_profile:
        profiler = Profile()
        profiler.enable()
    pm1, info = mat2pm(m1)
    m2, info = pm2mat(pm1)
    if do_profile:
        profiler.disable()
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()

    pm2, info = mat2pm(m2)
    assert_equal(len(pm1), 2 ** n - 1)
    assert_equal(len(pm2), 2 ** n - 1)
    assert_array_almost_equal(pm1, pm2, decimal=7)


def test_pm_info_to_string():
    info = PMInfo()
    info.smallest_pivot = 3.14
    info.number_of_times_ppivoted = 42
    s = pm_info_to_string(info)
    assert_equal(s, 'Pseudo-pivoted 42 times, smallest pivot used: 3.14')
    info.warn_not_odf = True
    info.warn_under_determined = True
    info.warn_inconsistent = True
    s = pm_info_to_string(info)
    assert_equal(s, '\
Pseudo-pivoted 42 times, smallest pivot used: 3.14;  \
pm2mat: off diagonal zeros found, solution suspect.;  \
pm2mat: multiple solutions to make rank(L-R)=1, solution suspect.;  \
pm2mat: input principal minors may be inconsistent, solution suspect.')
