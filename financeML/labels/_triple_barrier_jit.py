import numpy as np
from numba import njit, prange

@njit
def _compute_barrier_level(price: float, step: float, tp_val: float, sl_val: float, side: int):
    tp_active = tp_val > 0
    if side == 1:  # Long
        tp = price + tp_val * step if tp_active else np.nan
        sl = price - sl_val * step
    else:  # Short
        tp = price - tp_val * step if tp_active else np.nan
        sl = price + sl_val * step
    return tp, sl, tp_active

@njit
def _compute_barrier_hit_with_tp(price: np.ndarray, tp: float, sl: float, side: int):
    TP_HIT = 2
    UNCLOSED = 1
    SL_HIT = 0 

    T = len(price)
    if np.isnan(price[0]) or np.isnan(tp) or np.isnan(sl):
        return np.nan, np.nan

    for t in range(1, T):
        if side == 1:
            if price[t] >= tp:
                return t, TP_HIT
            elif price[t] <= sl:
                return t, SL_HIT
        else:  # side == -1
            if price[t] <= tp:
                return t, TP_HIT
            elif price[t] >= sl:
                return t, SL_HIT

    return T - 1, UNCLOSED

@njit
def _compute_barrier_hit_without_tp(price: np.ndarray, sl: float, side: int):
    TP_HIT = 2
    UNCLOSED = 1
    SL_HIT = 0

    T = len(price)
    if np.isnan(price[0]) or np.isnan(sl):
        return np.nan, np.nan

    for t in range(1, T):
        if side == 1:
            if price[t] <= sl:
                return t, SL_HIT
        else:  # side == -1
            if price[t] >= sl:
                return t, SL_HIT

    if side == 1:
        return T - 1, TP_HIT if price[-1] > price[0] else UNCLOSED
    else:
        return T - 1, TP_HIT if price[-1] < price[0] else UNCLOSED

@njit
def _compute_barrier_hit(price: np.ndarray, tp: float, sl: float, tp_active: bool, side: int):
    if tp_active:
        return _compute_barrier_hit_with_tp(price, tp, sl, side)
    else:
        return _compute_barrier_hit_without_tp(price, sl, side)

@njit
def _calculate_return(price: np.ndarray, t1: float, hit_type: float, tp: float, sl: float, use_tpsl: bool, side: int) -> float:
    if np.isnan(t1) or np.isnan(hit_type):
        return np.nan

    t1 = int(t1)
    hit_type = int(hit_type)

    if hit_type == 1:  # UNCLOSED
        exit_price = price[-1]
    elif use_tpsl:
        exit_price = tp if hit_type == 2 else sl
    else:
        exit_price = price[t1]

    if side == 1:
        return exit_price / price[0] - 1
    else:
        return price[0] / exit_price - 1

@njit
def _post_process_hit(hit_type: float, ret: float, neutral_class: bool, ret_thresh: float):
    if np.isnan(hit_type):
        return np.nan

    hit_type = int(hit_type)

    if neutral_class:
        return hit_type
    else:
        return 1 if ret > ret_thresh else 0

@njit
def _evaluate_event_instance(price: np.ndarray, step: float, tpsl: tuple, use_tpsl: bool, neutral_class: bool, ret_thresh: float, side: int):
    tp_val, sl_val = tpsl
    tp, sl, tp_active = _compute_barrier_level(price[0], step, tp_val, sl_val, side)
    t1, hit_type = _compute_barrier_hit(price, tp, sl, tp_active, side)
    ret = _calculate_return(price, t1, hit_type, tp, sl, use_tpsl, side)
    label = _post_process_hit(hit_type, ret, neutral_class, ret_thresh)
    return t1, label, ret

@njit(parallel=True)
def _evaluate_event(prices: np.ndarray, steps: np.ndarray, 
                    tpsl: tuple, use_tpsl: bool, neutral_class: bool, ret_thresh: float, sides: np.ndarray):
    N = prices.shape[0]
    t1s = np.full(N, np.nan)
    labels = np.full(N, np.nan)
    rets = np.full(N, np.nan)

    for i in prange(N):
        price = prices[i]
        step = steps[i]
        t1, label, ret = _evaluate_event_instance(price, step, tpsl, use_tpsl, neutral_class, ret_thresh, sides[i])
        t1s[i] = t1
        labels[i] = label
        rets[i] = ret

    return t1s, labels, rets





