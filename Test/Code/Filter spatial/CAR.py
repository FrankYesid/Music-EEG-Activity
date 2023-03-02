#  common average reference (CAR)
# ------------------------------------------------------------------------
# Version python
# Frank Y. Zapata C.
# Version 2021
# ------------------------------------------------------------------------
import numpy as np

def CAR(data):
    """
    The common average reference (CAR) was computed
    according to the formula  V_i: V_i-1/n sum 1to n V_j"
    where V_i is the potential between the ith electrode and the
    reference, and n is the number of electrodes in the montage
    (i.e. 64)
    """
    for i, tr in enumerate(data):
      prom    = tr.mean(axis=0)
      for ch in range(tr.shape[0]):
        data[i] = tr[ch,:] - prom

    return data
