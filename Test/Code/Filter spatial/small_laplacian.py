# Copyright by Clemens Brunner, Robert Leeb, Alois Schlögl
# Version matlab Toolbox Biosig
# ------------------------------------------------------------------------
# Version python
# Frank Y. Zapata C.
# Version 2020
# ------------------------------------------------------------------------

from .masker import generate_mask
import numpy as np

def small_laplacian(data, channels, montage_name):
    """Small laplacian
    Calculates spatial filter matrix for Laplacian derivations.
    ##
    Returns a spatial filter matrix used to calculate Laplacian derivations as
    well as indices used to plot in a topographical layout.
    Assuming that the data vector s is of dimension <samples x channels>, the
    Laplacian derivation s_lap can then be calculated by s_lap = s * lap.
    s = signal(trials x channels x time).
    ##
    Usage:
    # [lap, plot_index, n_rows, n_cols] = getMontage(montage)
    ##
    Input parameters:
    montage , Matrix containing the topographical layout of the channels. The
    content of this matrix can be one of the following formats:
    (1) Channel numbers where channels are located and zeros
    elsewhere <NxM>
    (2) Ones where channels are located and zeros elsewhere <NxM>
    (3) Predefined layout <string>.
    Examples for each format:
    (1) montage = [0 3,0 ,
    4,1 2 ,
    ,0 5,0]
    (2) montage = [0,1,0 ,
    ,1,1,1 ,
    ,0,1,0]
    (3) montage = '16ch'
    ##
    Output parameters:
    lap        , Laplacian filter matrix
    plot_index , Indices for plotting the montage
    n_rows     , Number of rows of the montage
    n_cols     , Number of columns of the montage
    """
    montage = generate_mask(channels, montage_name)
    temp = montage

    counter = 1
    temp = np.asarray(temp).T
    lap = np.zeros((np.size(temp, 0), np.size(temp, 1)))

    # Used electrode positions instead of ones (format (1))
    positions = []
    if sum(sum(temp)) != (sum(sum(temp > 0))):
        tmp = np.sort(temp[np.where(temp)])
        positions = np.argsort(temp[np.where(temp)])
        temp = temp > 0

    for k1 in range(0, np.size(temp, 1)):
        for k2 in range(0, np.size(temp, 0)):
            if temp[k2, k1] >= 1:
                lap[k2, k1] = counter
                counter = counter + 1

    neighbors = np.empty((counter - 1, 4))
    neighbors[:] = np.nan

    lap_ = np.zeros((lap.size))
    a = 0
    for k1 in range(0, np.size(temp, 1)):
        for k2 in range(0, np.size(temp, 0)):
            lap_[a] = lap[k2][k1]
            a += 1

    neighbors = []
    for col, row in np.ndindex(lap.T.shape):
        if lap[row][col]:
            around = np.array(
                [row + 1, row - 1, col + 1, col - 1], dtype=object)
            pair = np.array([col, col, row, row], dtype=object)

            around[around < 0] = None

            if around[0] >= lap.shape[0]:
                around[0] = None

            if around[2] >= lap.shape[1]:
                around[2] = None

            around[2:], pair[2:] = pair[2:], around[2:].copy()
            indexes = filter(lambda c: None not in c, zip(around, pair))

            neighbors.append(
                ([lap[ind] for ind in indexes if lap[ind]] + [0] * 4)[:4])

    neighbors = np.array(neighbors, dtype=object)
    # neighbors[neighbors==0] = np.nan # colocar los datos en NaN en ceros 0.

    lap = np.eye(neighbors.shape[0], dtype=float)

    for k in range(0, neighbors.shape[0]):
        temp = neighbors[k, neighbors[k, :] != 0].astype(
            int)  # Neighbors of electrode k
        for aa in range(0, len(temp)):
            lap[k, temp[aa] - 1] = -1 / temp.shape[0]

    if not len(positions):
        lap = lap[positions, positions]

    lap = lap.T
    for i, tr in enumerate(data):
        data[i] = lap.dot(tr)

    return data
