import numpy as np
from numba import njit,prange

@njit
def _find_step_1d(price: np.ndarray) -> float:
    """
    ----------
    prices : (T)
    """
    log_return = np.log(price[1:] / price[:-1])
    avg_price = np.nanmean(price)
    step = avg_price*np.nanstd(log_return)
    return step

@njit(parallel = True)
def _find_step_1d_window(price: np.ndarray, window = 64) -> np.ndarray:
    """
    ----------
    prices : (T)
    """
    T = price.shape
    step = np.full(T, np.nan)
    for t in prange(window - 1, len(price)):
        price_kernal = price[t - window + 1 : t + 1]
        step[t] = _find_step_1d(price_kernal)
    return step


@njit(parallel = True)
def _find_step_2d_window(prices: np.ndarray, window = 64) -> np.ndarray:
    """
    ----------
    prices : (T,S)
    """
    T,S = prices.shape
    step_2d = np.full((T,S), np.nan)
    for s in prange(S):
        price = prices[:,s]
        for t in prange(window - 1, len(price)):
            price_kernal = price[t - window + 1 : t + 1]
            step_2d[t,s] = _find_step_1d(price_kernal)
    return step_2d