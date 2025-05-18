import numpy as np
import pandas as pd
from typing import Tuple

from financeML.util import get_event_signal, compute_step
from financeML.labels._triple_barrier_jit import _evaluate_event

def evaluate_trade_result(
    event_df: pd.DataFrame,
    price_df: pd.DataFrame,
    step_df: pd.DataFrame | None = None,
    lookback: int = 64,
    tpsl: Tuple[float, float] | None = None,
    use_tpsl: bool = False,
    neutral_class: bool = True,
    ret_thresh: float = 0.01,
    side: int | None = 1, 
    window: int = 10,
    trade_next: bool = True
) -> pd.DataFrame:
    """
    Evaluate trade outcomes for event-aligned trades using the triple barrier method.

    Parameters
    ----------
    event_df : pd.DataFrame
        Must include:
        - 'date': pd.Timestamp of the event signal
        - 'stock_id': security identifier matching columns in `price_df`
        - 'side' (optional): +1 for long, -1 for short. Required if `side=None`.

    price_df : pd.DataFrame
        Price matrix with:
        - Index: timestamps
        - Columns: stock IDs
        - Values: float prices (e.g., closing prices)

    step_df : pd.DataFrame
        Matrix of step values (e.g., volatility, atr) used to scale the take-profit and stop-loss levels.
        Must be same shape and index/column-aligned with `price_df`.

        If None, compute by volatility
    
    lookback: int, default=64
        Only used when `step_df=None`. Number of past timesteps to calculate rolling volatility

    tpsl : tuple[float, float] | None, default=None
        Take-profit and stop-loss multipliers applied to the `step` size (e.g., volatility or ATR):

        - For long trades:  
          - TP barrier = `price + tp * step`
          - SL barrier = `price - sl * step`
        - For short trades:  
          - TP barrier = `price - tp * step`
          - SL barrier = `price + sl * step`

        Use `tp = 0` to disable take-profit evaluation (e.g., `tpsl=(0, 2)`).

        If set to `None` and `step_df` is also `None`, the default is dynamically computed as:
        `tpsl = (√window, √window)`, which aligns with the expected return distribution under Brownian motion.
    
    use_tpsl : bool, default=False
        If True: use the actual TP/SL levels as the exit price when a barrier is hit.
        If False: use the price at `exit_date` for exit.

    neutral_class : bool, default=True
        If True: labels represent barrier outcomes as:
            - 0 = SL hit
            - 1 = Unclosed
            - 2 = TP hit  
        If False: labels are binary:
            - 1 = return > `ret_thresh`
            - 0 = return <= `ret_thresh`

    ret_thresh : float, default=0.01
        Only used when `neutral_class=False`. Sets the minimum return to be considered profitable.

    side : int or None, default=1
        Direction of trade:
        - If int (+1 or -1), applies the same direction to all events.
        - If None, uses the `side` column from `event_df`.

    window : int, default=10
        Number of future timesteps to monitor the trade for barrier events.

    trade_next : bool, default=True
        If True, assumes trade is executed at price(t+1) after observing features at price(t).
        If False, assumes trade is executed immediately at price(t).

    Returns
    -------
    label_df : pd.DataFrame
        Labeled trade outcomes with columns:
        - 'stock_id': from `event_df`
        - 'entry_date': Timestamp when trade is entered (t or t+1)
        - 'exit_date': Timestamp when trade is exited (TP, SL, or final window)
        - 'label': classification label (multi-class or binary)
        - 'return': realized return over the trade
        - 'side': long (+1) or short (-1)

    Notes
    -----
    - Use `neutral_class=False` when training a binary classifier to predict profitable trades.
    - Use `neutral_class=True` when learning the type of barrier event (TP, SL, or unclosed).
    - The barrier labeling is aligned with the signal window. The price window used for evaluation
      starts at `price(t+1)` if `trade_next=True`, otherwise from `price(t)`.
    - To disable the take-profit condition and only evaluate stop-loss (or final return),
      set `tp = 0` in the `tpsl` parameter (e.g., `tpsl=(0, 2)`).
    - If `tpsl=None` and `step_df=None`, the take-profit and stop-loss thresholds default to
      `(√window, √window)`. This choice reflects the empirical behavior of returns under
      Brownian motion (`σ√T`) and is intended to generate balanced class labels across TP, SL, and unclosed.
    - If `step_df` is explicitly provided, `tpsl` must also be explicitly set to avoid ambiguity.
    - The `use_tpsl` parameter controls whether the exit price is set to the barrier level (TP/SL),
      or the actual price at the time the barrier is triggered. Default is `False` to reflect
      realistic execution, since trades may not occur at the exact threshold price in illiquid or
      discrete markets.
    """

    # handle tpsl
    if tpsl is None:
        if step_df is None:
            tpsl = (np.sqrt(window), np.sqrt(window))
        else:
            raise ValueError("tpsl must be explicitly set if step_df is provided.")

    # get aligned signal windows
    if step_df is None:
        step_df = compute_step(price_df, lookback)
        
    price_extractor = get_event_signal(event_df, price_df)
    step_extractor = get_event_signal(event_df, step_df)
    prices = price_extractor[1:window+1] if trade_next else price_extractor[:window]  # (N, window) 
    steps = step_extractor[0] # (N,)

    N = len(event_df)

    # handle side
    if side is None:
        if "side" not in event_df.columns:
            raise ValueError("`side` is None and 'side' column not found in event_df.")
        sides = event_df["side"].astype(np.int32).values
    else:
        sides = np.full(N, side, dtype=np.int32)

    # evaluate barrier outcome
    t1s, labels, rets = _evaluate_event(prices, steps, tpsl, use_tpsl, neutral_class, ret_thresh, sides) # (N), (N), (N)
    
    # compute entry index based on trade_next
    date_index = price_df.index
    event_idx = date_index.get_indexer(event_df['date'])
    entry_idx = event_idx + 1 if trade_next else event_idx # (N)

    # handle t1 = nan
    nan_mask = np.isnan(t1s)
    t1s_int = np.where(nan_mask, -1, t1s).astype(np.int32)
    exit_idx = entry_idx + t1s_int

    # bounds check & t1 validation
    valid_entry = (entry_idx >= 0) & (entry_idx < len(date_index))
    valid_exit = (exit_idx >= 0) & (exit_idx < len(date_index)) & ~nan_mask

    entry_dates = np.full(N, np.datetime64('NaT'), dtype='datetime64[ns]')
    exit_dates = np.full(N, np.datetime64('NaT'), dtype='datetime64[ns]')
    entry_dates[valid_entry] = date_index.values[entry_idx[valid_entry]]
    exit_dates[valid_exit] = date_index.values[exit_idx[valid_exit]]

    return pd.DataFrame({
        "stock_id": event_df["stock_id"].values,
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "label": labels,
        "return": rets,
        "side": sides
    })