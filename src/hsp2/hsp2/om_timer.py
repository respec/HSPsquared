"""
The class timer_class/timer_class_jit is used for benchmarking and peformance inquiry.
"""

from numba.experimental import jitclass
import ctypes
import time
# Access the _PyTime_AsSecondsDouble and _PyTime_GetSystemClock functions from pythonapi
get_system_clock = ctypes.pythonapi._PyTime_GetSystemClock
as_seconds_double = ctypes.pythonapi._PyTime_AsSecondsDouble
# Set the argument types and return types of the functions
get_system_clock.argtypes = []
get_system_clock.restype = ctypes.c_int64
as_seconds_double.argtypes = [ctypes.c_int64]
as_seconds_double.restype = ctypes.c_double

timer_spec = [
     ("tstart", types.float64),
     ("tend", types.float64),
     ("tsplit", types.float64)
]

@njit
def jitime():
    system_clock = get_system_clock()
    current_time = as_seconds_double(system_clock)
    return current_time

@jitclass(timer_spec)
class timer_class_jit():
    def __init__(self):
        self.tstart = jitime()
    
    def split(self):
        self.tend = jitime()
        self.tsplit = self.tend - self.tstart
        self.tstart = jitime()
        split = 0
        if (self.tsplit > 0):
            split = self.tsplit
        return split

class timer_class():
    def __init__(self):
        self.tstart = time.time()
    
    def split(self):
        self.tend = time.time()
        self.tsplit = self.tend - self.tstart
        self.tstart = time.time()
        split = 0
        if (self.tsplit > 0):
            split = self.tsplit
        return split
