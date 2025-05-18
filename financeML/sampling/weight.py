import numpy as np
import pandas as pd
from ._weight_jit import _calculate_avgU_and_SW_multiasset

def compute_event_weights(label_df:pd.DataFrame, price_df:pd.DataFrame, weight_by_return: bool = True):
    """
    Compute uniqueness and sample weights for labeled trading events.

    Parameters
    ----------
    label_df : pd.DataFrame
        Must contain: 'stock_id', 'entry_date', 'exit_date' (as pd.Timestamp).
    
    price_df : pd.DataFrame
        Rows = trading dates, Columns = stock IDs.
        Must cover full range of entry/exit dates.

    weight_by_return : bool, default=True
        If True, weight sample by concurrency-adjusted return.

    Returns
    -------
    weight_df : pd.DataFrame
        Index: aligned with label_df
        Columns:
        - 'uniqueness'
        - 'sample_weight'
    """

    weight_df = pd.DataFrame(columns=["uniqueness", "sample_weight"], index=label_df.index)
    valid_labels = label_df.dropna()

    date_to_index = {date: idx for idx, date in enumerate(price_df.index)}
    stock_to_index = {stock: idx for idx, stock in enumerate(price_df.columns)}

    entry_idx = valid_labels['entry_date'].map(date_to_index)
    exit_idx = valid_labels['exit_date'].map(date_to_index) + 1 # +1 to include exit date to concurrency count
    stock_idx = valid_labels['stock_id'].map(stock_to_index)

    time_span_array = np.stack([entry_idx, exit_idx, stock_idx], axis=1)

    if weight_by_return:
        return_df = price_df.pct_change(fill_method=None)
        returns = return_df.values
    else:
        returns = np.ones(price_df.shape)

    avgU_and_SW = _calculate_avgU_and_SW_multiasset(time_span_array, returns)
    weight_df.loc[valid_labels.index] = avgU_and_SW

    return weight_df