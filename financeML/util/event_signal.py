import numpy as np
import pandas as pd

class EventWindowExtractor:
    def __init__(self, signal: np.ndarray, t_idx: np.ndarray, s_idx: np.ndarray):
        self.signal = signal
        self.t_idx = t_idx
        self.s_idx = s_idx
        self.N = len(t_idx)
        self.T = signal.shape[0]  

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._extract_single_offset(key)
        elif isinstance(key, slice):
            offsets = list(range(
            key.start if key.start is not None else 0,
            key.stop if key.stop is not None else 0,
            key.step if key.step is not None else 1
            ))
            return self._extract_multi_offset(offsets)
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def _extract_single_offset(self, offset: int) -> np.ndarray:
        indices = self.t_idx + offset
        valid = (indices >= 0) & (indices < self.T)
        out = np.full(self.N, np.nan, dtype=np.float64)

        # Only assign valid entries
        out[valid] = self.signal[indices[valid], self.s_idx[valid]]
        return out

    def _extract_multi_offset(self, offsets: list[int]) -> np.ndarray:
        N, W = self.N, len(offsets)
        out = np.full((N, W), np.nan, dtype=np.float64)

        for j, offset in enumerate(offsets):
            indices = self.t_idx + offset
            valid = (indices >= 0) & (indices < self.T)

            # Assign only valid values to current window column
            out[valid, j] = self.signal[indices[valid], self.s_idx[valid]]
        return out

def get_event_signal(event_df: pd.DataFrame, signal_df: pd.DataFrame) -> EventWindowExtractor:
    """
    Aligns event times and stock IDs from an event DataFrame with a signal DataFrame,
    and returns an extractor that supports slicing signal values relative to each event.

    Parameters
    ----------
    event_df : pd.DataFrame
        A DataFrame containing at least two columns:
        - 'date': datetime values representing the event times
        - 'stock_id' : identifiers matching the signal_df columns (as strings or ints)

    signal_df : pd.DataFrame
        A time-series signal matrix with:
        - index: datetime (must cover all event_df['date'])
        - columns: stock IDs (as strings or ints)
        - values: signal values (e.g., price, volume)

    Returns
    -------
    EventWindowExtractor
        An object that supports slicing and indexing like a NumPy array.
        - extractor[0] returns the signal at event time
        - extractor[-3:] returns the 3-timestep window *before* each event
        - extractor[1:6] returns a 5-timestep window *after* each event

    Notes
    -----
    - The signal is automatically cast to float64 to support np.nan for out-of-bound indices.
    - If an event's (timestamp, stock_id) is outside the signal_df range, np.nan is returned.
    - Useful for computing event-aligned features (e.g., moving averages, RSI, etc.)

    Example
    -------
    >>> event_signal = get_event_signal(event_df, close_df)
    >>> x_now = event_signal[0]       # signal at event time
    >>> x_fut = event_signal[1:6]     # next 5 time steps
    >>> x_past = event_signal[-3:0]   # 3 timesteps before the event
    """
    # Convert index and column mapping
    date_to_idx = {d: i for i, d in enumerate(signal_df.index)}
    stockid_to_idx = {sid: i for i, sid in enumerate(signal_df.columns)}

    try:
        t_idx = np.array([date_to_idx[ts] for ts in event_df["date"]])
        s_idx = np.array([stockid_to_idx[sid] for sid in event_df["stock_id"]])
    except KeyError as e:
        raise ValueError(f"[Event Mapping Error] {e} not found in signal_df index or columns.")

    return EventWindowExtractor(signal_df.values, t_idx, s_idx)
