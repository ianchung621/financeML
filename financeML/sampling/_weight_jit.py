import numpy as np
from numba import njit

@njit
def _calculate_avgU_and_SW_multiasset(time_span_array: np.ndarray, returns: np.ndarray):
    """fast computation of average uniqueness and sample weights

    Parameters
    ----------
    time_span_array : np.ndarray (N,3)
        date_idx, t1_idx, stock_id of N events
        stock_id must maps to the col index of returns
    returns : np.ndarray (T, S)
        returns of trading_date
    
    Returns
    --------
    avgU_and_SW: np.ndarray (N,2)
        average uniqueness, sample weights of N events
    """

    N = time_span_array.shape[0]
    T = returns.shape[0]
    label_concurrency = np.zeros(T)  # concurrency at each time

    entry_idx = time_span_array[:, 0]
    exit_idx = time_span_array[:, 1]
    stock_idx = time_span_array[:, 2]

    avg_U = np.zeros(N)
    SW = np.zeros(N)

    for i in range(N):
        label_concurrency[entry_idx[i]:exit_idx[i]] += 1

    for i in range(N):

        e = entry_idx[i]
        x = exit_idx[i]
        s = stock_idx[i]
        lc_slice = label_concurrency[e:x]
        ret_slice = returns[e:x, s]

        avg_U[i] = np.mean(1 / lc_slice)
        SW[i] = np.sum(ret_slice / lc_slice)

    SW = np.abs(SW)
    SW = len(SW[~np.isnan(SW)]) * SW/np.nansum(SW) # scale the SW that its sum is not-nan row count
    
    avgU_and_SW = np.zeros((N,2))
    avgU_and_SW[:,0] = avg_U
    avgU_and_SW[:,1] = SW
    return avgU_and_SW