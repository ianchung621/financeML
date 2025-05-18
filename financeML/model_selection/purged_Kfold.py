import numpy as np
import pandas as pd

class PurgedKFold:
    """
        Implements Purged K-Fold Cross-Validation for time series and financial data.

        This cross-validation method ensures that the training and test sets do not overlap in 
        a way that introduces look-ahead bias. It achieves this by:
        1. Splitting the data into `n_splits` folds based on trading dates.
        2. Applying an embargo period before and after each test set to prevent data leakage.
        3. Returning train-test splits where the training set excludes data within the embargo period.

        Attributes:
        -----------
        trading_date : pd.DatetimeIndex
            A sorted index of trading dates to be used for partitioning.
        embargo_days : int
            The number of trading days to exclude before and after each test set.
        n_splits : int, optional (default=5)
            The number of folds to generate.

        Methods:
        --------
        split(df: pd.DataFrame)
            Generates train-test splits while ensuring no data leakage.
        Example:
        --------
        Given:
            - trading dates: [0, 1, 2, ...]
            - embargo_days = 3
            - One of the test sets: [13, 14, 15, 16]

        The embargo period:
            - Pre-test embargo: [..., 7, 8, 9]
            - Post-test embargo: [20, 21, 22, ...]

        Resulting train-test split:
            train = [..., 7, 8, 9, 20, 21, 22, ...]
            test  = [13, 14, 15, 16]

        Notice:
            - The **gap** between `train` and `test` is `embargo_days + 1 = 4`.
            - Training set **does not contain** any test dates or embargoed dates.
        """
    def __init__(self, trading_date:pd.DatetimeIndex, embargo_days:int, n_splits:int = 5):
        
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.trading_date = trading_date if trading_date.is_monotonic_increasing else trading_date.sort_values()
    
    def split(self, event_df: pd.DataFrame, label_df: pd.DataFrame):
        """
        Parameters
        ----------
        event_df : pd.DataFrame
            Must contain 'date', aligned with label_df (index must match)

        label_df : pd.DataFrame
            Must contain 'label'. Any NaN will be excluded from train/test.

        Yields
        -------
        train_idx, test_idx : np.ndarray

        """
        assert event_df.index.equals(label_df.index), "Index mismatch between event_df and label_df"

        valid_mask = label_df.notna().all(axis=1)
        event_df = event_df.loc[valid_mask]
        label_df = label_df.loc[valid_mask]

        # trim trading_date based on event_df
        start_idx = np.searchsorted(self.trading_date, event_df['date'].min(), side='left')
        end_idx = np.searchsorted(self.trading_date, event_df['date'].max(), side='right')
        trading_date = self.trading_date[start_idx:end_idx] 

        T = len(trading_date)
        K = self.n_splits
        mbrg = self.embargo_days
        trading_date_idx = np.arange(T)
        test_splits = np.array_split(trading_date_idx, K)
        for test_idxs in test_splits:
            test_start, test_end = test_idxs[0], test_idxs[-1] + 1 # +1 for slicing
            pre_test_embargo = max(0, test_start - mbrg)
            post_test_embargo = min(T, test_end + mbrg)
            
            train_mask = (event_df['date'].isin(trading_date[:pre_test_embargo]) |
                          event_df['date'].isin(trading_date[post_test_embargo:]))

            test_mask = event_df['date'].isin(trading_date[test_idxs])

            train_idx = label_df.index[train_mask].to_numpy()
            test_idx = label_df.index[test_mask].to_numpy()

            yield train_idx, test_idx