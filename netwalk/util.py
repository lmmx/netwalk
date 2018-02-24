# convenience functions to be imported by other modules
from datetime import datetime as dt

def ptime():
    """
    Print the current time (hours:minutes:seconds), for debug time profiling.
    """
    t_now = dt.now()
    t = f"{t_now.hour}:{t_now.minute}-{t_now.second}s:{t_now.microsecond}"
    return t
