import numpy as np
from scipy.io import wavfile


def signaltonoise(filename):
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio, defined as the mean
    divided by the standard deviation.
    """
    sc = np.sum(wavfile.read(filename)[1], axis=1)
    norm = sc / (max(np.amax(sc), -1 * np.amin(sc)))
    a = np.asanyarray(norm)
    m = a.mean(0)
    sd = a.std(axis=0, ddof=0)
    return float(np.where(sd == 0, 0, m / sd))
