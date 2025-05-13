# util/step.py
import numpy as np
import pandas as pd
from typing import Union
from ._find_step_jit import _find_step_1d_window, _find_step_2d_window

def compute_step(
    price: Union[pd.Series, pd.DataFrame], 
    window: int = 64
) -> Union[pd.Series, pd.DataFrame]:
    """
    Estimate adaptive step size based on log return volatility in a rolling window.

    Parameters
    ----------
    price : pd.Series or pd.DataFrame
        Price series (T,) or matrix (T, S)
    window : int
        Rolling window length. Must be >= 2.

    Returns
    -------
    pd.Series or pd.DataFrame
        Step size series or matrix, aligned with input index and columns
    """
    if not isinstance(price, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame")

    if window < 2:
        raise ValueError("`window` must be >= 2")

    if isinstance(price, pd.Series):
        price_arr = price.to_numpy(dtype=np.float64)
        step_arr = _find_step_1d_window(price_arr, window)
        return pd.Series(step_arr, index=price.index, name=price.name)
    
    elif isinstance(price, pd.DataFrame):
        price_arr = price.to_numpy(dtype=np.float64)
        step_arr = _find_step_2d_window(price_arr, window)
        return pd.DataFrame(step_arr, index=price.index, columns=price.columns)
