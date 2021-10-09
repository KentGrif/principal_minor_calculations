import numpy as np
from nose.tools import assert_equal, assert_in
from numpy.testing import assert_array_equal, assert_array_almost_equal
from principal_minor_calculations import mat2pm, pm2mat, _msb
from cProfile import Profile
from pstats import Stats


def test_msb():
    assert_equal(_msb(1), 1)
    assert_equal(_msb(2), 2)
    assert_equal(_msb(7), 4)
    assert_equal(_msb(6), 4)
    assert_equal(_msb(13), 8)
    assert_equal(_msb(0x1234567812345678), 0x1000000000000000)
    assert_equal(_msb(0xf234567812345678), 0x8000000000000000)


def test_mat2pm():
    m1 = np.array([[1, 2],
                   [3, 4]])
    pm1, msg = mat2pm(m1)
    assert_array_equal(pm1, [1, 4, -2])
    m2, msg = pm2mat(pm1)
    pm2, msg = mat2pm(m2)
    assert_array_equal(pm1, pm2)

    m1 = np.array([[1, 2, 6],
                   [2, 4, 5],
                   [-1, 2, 3]])
    pm1, msg = mat2pm(m1)
    assert_array_equal(pm1, [1, 4, 0, 3, 9, 2, 28])
    m2, msg = pm2mat(pm1)
    pm2, msg = mat2pm(m2)
    assert_array_almost_equal(pm1, pm2, decimal=10)

    m1 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0]])
    pm1, msg = mat2pm(m1)
    assert_array_equal(pm1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    assert_equal(msg, 'mat2pm: pseudo-pivoted 7 times, smallest pivot used: 0.25')
    m2, msg = pm2mat(pm1)
    assert_array_equal(m2, [[0, 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0]])
    assert_in('off diagonal zeros found', msg)
    assert_in('multiple solutions to make rank', msg)
    pm_inconsistent = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    m, msg = pm2mat(pm_inconsistent)
    assert_in('principal minors may be inconsistent', msg)
    pm, msg = mat2pm(m)
    # all but two of the principal minors actually match
    pm[12] = 0
    pm[14] = 0
    pm_inconsistent[12] = 0
    pm_inconsistent[14] = 0
    assert_array_almost_equal(pm, pm_inconsistent, decimal=10)

    m1 = np.array([[-6, 3, -9, 4],
                   [-6, -5, 3, 6],
                   [3, -3, 6, -7],
                   [1, 1, -1, -3]])
    pm1, msg = mat2pm(m1)
    pm_truth = np.array([-6, -5, 48, 6, -9, -21, -36, -3, 14, 9, -94, -25, 96, 59, 6])
    assert_equal(msg, 'mat2pm: pseudo-pivoted 0 times, smallest pivot used: 0.75')
    assert_array_almost_equal(pm1, pm_truth, decimal=10)
    m2, msg = pm2mat(pm_truth)
    pm2, msg = mat2pm(m2)
    assert_array_almost_equal(pm1, pm2, decimal=10)

    n = 18
    m1 = np.random.rand(n, n)

    do_profile = False
    if do_profile:
        profiler = Profile()
        profiler.enable()
    pm1, msg = mat2pm(m1)
    m2, msg = pm2mat(pm1)
    if do_profile:
        profiler.disable()
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()

    pm2, msg = mat2pm(m2)
    assert_equal(len(pm1), 2 ** n - 1)
    assert_equal(len(pm2), 2 ** n - 1)
    assert_array_almost_equal(pm1, pm2, decimal=8)
