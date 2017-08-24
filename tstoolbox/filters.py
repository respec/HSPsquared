"""
The filter.py module contains a group of functions to filter
time-series values.
"""

from __future__ import print_function

#===imports======================
import numpy as np

#===globals======================
modname = "filter"

########
# Exception classes


class MisMatchedKernel(Exception):
    '''
    Error class for the wrong length kernel.
    '''
    def __init__(self, rk, pw):
        self.rk = rk
        self.pw = pw

    def __str__(self):
        return """
*
*   Length of kernel must be %i.
*   Instead have %i
*
""" % (self.rk, self.pw)


class BadKernelValues(Exception):
    '''
    Error class for the negative pad width.
    '''
    def __init__(self):
        pass

    def __str__(self):
        return """
*
*   Should only have positive values.
*
"""


def _transform(vector, cutoff_period, window_len, lopass=None):
    """

    Parameters
    ----------
    vector : array_like, evenly spaced samples in time

    Returns
    -------
    vector of filtered values

    See Also
    --------

    Examples
    --------

    """
    if cutoff_period is None:
        raise ValueError('''
*
*   The cutoff_period must be set.
*
''')

    if window_len is None:
        raise ValueError('''
*
*   The window_len must be set.
*
''')

    import numpy.fft as F
    result = F.rfft(vector, len(vector))

    freq = F.fftfreq(len(vector))[:len(vector)/2 + 1]
    factor = np.ones_like(freq)

    if lopass is True:
        factor[freq > 1.0/float(cutoff_period)] = 0.0
        factor = np.pad(factor, window_len + 1, mode='constant',
                        constant_values=(1.0, 0.0))
    else:
        factor[freq < 1.0/float(cutoff_period)] = 0.0
        factor = np.pad(factor, window_len + 1, mode='constant',
                        constant_values=(0.0, 1.0))

    factor = np.convolve(factor, [1.0/window_len]*window_len, mode=1)
    factor = factor[window_len + 1:-(window_len + 1)]

    result = result * factor

    rvector = F.irfft(result, len(vector))

    return np.atleast_1d(rvector)


if __name__ == '__main__':
    ''' This section is just used for testing.  Really you should only import
        this module.
    '''
    arr = np.arange(100)
    print(arr)
    print(np.median(arr, (3, )))
    print(np.constant(arr, (-25, 20), (10, 20)))
    arr = np.arange(30)
    arr = np.reshape(arr, (6, 5))
    print(np.mean(arr, pad_width=((2, 3), (3, 2), (4, 5)), stat_len=(3, )))
