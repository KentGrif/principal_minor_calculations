import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
from principal_minor_calculations import mat2pm, _msb
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
    m = np.array([[1, 2],
                  [3, 4]])
    pm = mat2pm(m)
    assert_array_equal(pm, [1, 4, -2])

    m = np.array([[1, 2, 6],
                  [2, 4, 5],
                  [-1, 2, 3]])
    pm = mat2pm(m)
    assert_array_equal(pm, [1, 4, 0, 3, 9, 2, 28])

    m = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0]])
    pm = mat2pm(m)
    assert_array_equal(pm, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

    n = 20
    m = np.random.rand(n, n)

    do_profile = False
    if do_profile:
        profiler = Profile()
        profiler.enable()
    pm = mat2pm(m)
    if do_profile:
        profiler.disable()
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()

    assert_equal(len(pm), 2**n - 1)
