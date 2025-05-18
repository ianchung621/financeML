# 📘 finance ML

Implementation of techniques from for [*Advances in Financial Machine Learning*](https://www.wiley.com/en-fr/Advances+in+Financial+Machine+Learning-p-9781119482109) by Marcos López de Prado

<p align="center">
  <img src="https://www.wiley.com/storefront-pdp-assets/_next/image?url=https%3A%2F%2Fmedia.wiley.com%2Fproduct_data%2FcoverImage300%2F89%2F11194820%2F1119482089.jpg&w=640&q=75" alt="Advances in Financial Machine Learning" width="200"/>
</p>

## 🗂️ Project Structure

Each submodule under **`financeML/`** implements a component of the financial ML workflows. User-facing API functions are exposed via each module’s `__init__.py`.

* **`labels/`**: Labeling logic using the triple barrier method
* **`sampling/`**: Computation of label concurrency-aware sample weights
* **`model_selection/`**: Purged K-Fold cross-validation to avoid temporal leakage
* **`features/`**: (To be implemented) feature engineering utilities
* **`backtest/`**: (To be implemented) trade outcome evaluation
* **`util/`**: Helper functions for computation

## 🛠️ Example Usage

### 1. Prepare `event_df` and `price_df`

These are the primary inputs to the labeling pipeline:

* `event_df`: contains trade-triggering events with columns:

  * `'date'`: event timestamp
  * `'stock_id'`: stock identifiers (e.g., `'GOOG'`, `'MSFT'`)
  * optional `'side'`: +1 for long, -1 for short
  * any additional features columns for model training

* `price_df`: a time-indexed DataFrame of prices:

  * Index: pd.Timestamp
  * Columns: stock identifiers
  * Values: float prices (typically close or adj-close)

#### Example

**`event_df`**

| date       | stock\_id | side | return\_5d | volatility\_10d |
| ---------- | --------- | ---- | ---------- | --------------- |
| 2025-01-02 | GOOG      | 1    | 0.012      | 0.015           |
| 2025-01-02 | MSFT      | -1   | -0.008     | 0.010           |

**`price_df`**

|            | GOOG  | MSFT  |
| ---------- | ----- | ----- |
| 2025-01-02 | 100.0 | 240.0 |
| 2025-01-03 | 101.5 | 238.0 |
| 2025-01-04 | 102.2 | 237.5 |
| ...        | ...   | ...   |

---

### 2. Run Triple Barrier Labeling

The **triple barrier method** is a labeling technique from *Advances in Financial Machine Learning*. It evaluates each event by monitoring price movements over a future window, labeling the outcome based on which of the following occurs first:

* **Take-profit (TP)**: price crosses the upper barrier → `label = 2`
* **Stop-loss (SL)**: price crosses the lower barrier → `label = 0`
* **Unclosed**: neither TP nor SL is hit → `label = 1`

The method supports both:

* **Multi-class classification** via `neutral_class=True` (labels ∈ {0, 1, 2})
* **Binary classification** via `neutral_class=False` and `ret_thresh`

```python
from financeML.labels import evaluate_trade_result
label_df = evaluate_trade_result(event_df, price_df)
```

**Function: `evaluate_trade_result`**

```python
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
```

**Notes**

* Use `neutral_class=False` for binary labels:

  * `1`: return > `ret_thresh`
  * `0`: return ≤ `ret_thresh`
* Use `neutral_class=True` for multi-class barrier labeling:

  * `2`: take-profit hit
  * `1`: unclosed within window
  * `0`: stop-loss hit
* If `tpsl` and `step_df` are both `None`, price barriers default to √window scaling
  `tpsl = (√window, √window)`, consistent with Brownian motion:
  Δprice ~ σ√T

**Returns**

The function outputs a `label_df` with the following columns:

| stock\_id | entry\_date | exit\_date | label | return | side |
| --------- | ----------- | ---------- | ----- | ------ | ---- |
| GOOG      | 2025-01-02  | 2025-01-06 | 2     | 0.042  | 1    |
| MSFT      | 2025-01-02  | 2025-01-05 | 0     | -0.031 | -1   |

* `entry_date`: trade entry timestamp (`date` or `date + 1` depending on `trade_next`)
* `exit_date`: timestamp when TP/SL was hit or window ended
* `label`: classification outcome (`0`, `1`, or `2` depending on config)
* `return`: realized return over the trade — computed using barrier price if `use_tpsl=True`, or market price at `exit_date` if `use_tpsl=False` (default, more realistic for discrete/illiquid markets)
* `side`: long (+1) or short (-1)

See docstring for detailed param usage in [financeML/labels/triple_barrier.py](financeML/labels/triple_barrier.py)

---

### 3. Compute Sample Weights

After labeling, we compute sample weights to account for overlapping events (concurrency) and, optionally, scale by realized returns. This is useful for training classifiers that are robust to label redundancy

```python
from financeML.sampling import compute_event_weights
weight_df = compute_event_weights(label_df, price_df)
```

**Function: `compute_event_weights`**

```python
def compute_event_weights(
    label_df: pd.DataFrame,
    price_df: pd.DataFrame,
    weight_by_return: bool = True
) -> pd.DataFrame:
```

**Returns**

A DataFrame aligned with `label_df` index:

| uniqueness | sample\_weight |
| ---------- | -------------- |
| 0.32       | 0.013          |
| 0.45       | 0.020          |

* `uniqueness`: measures how isolated each event is over time; lower concurrency → higher uniqueness
  (i.e., the harmonic average of overlapping events during the trade window)
* `sample_weight`: equals `uniqueness × |return|` if `weight_by_return=True`, otherwise equals `uniqueness` only

Use this to pass `sample_weight` during model training

See Implementation in [financeML/sampling/weight.py](financeML/sampling/weight.py)

---

### 4. Cross-Validation with Purged K-Fold

The `PurgedKFold` class helps prevent leakage in time-series classification by:

* **Purging:** Ensuring no temporal overlap between training and test events
* **Embargoing:** Applying an embargo period around each test set

**Initialization:**

```python
from financeML.model_selection import PurgedKFold
pkf = PurgedKFold(trading_date=price_df.index, embargo_days=100)
```

**Example Downstream Task (Model Training):**

```python
from sklearn.ensemble import RandomForestClassifier

for train_idx, test_idx in pkf.split(event_df, label_df):
    feat_col = [col for col in event_df.columns if col not in ['date', 'stock_id']]
    X_train = event_df.loc[train_idx, feat_col]
    y_train = label_df.loc[train_idx, 'label']
    X_test = event_df.loc[test_idx, feat_col]
    y_test = label_df.loc[test_idx, 'label']
    weights = weight_df.loc[train_idx]

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced_subsample',
        max_samples=weights['uniqueness'].mean(),
        random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=weights['sample_weight'])
```

See implementation in [`financeML/model_selection/purged_Kfold.py`](financeML/model_selection/purged_Kfold.py)
