"""
BCG X – CodeSignal DSF Complete Cheat Sheet
============================================
Covers all 4 modules:
  1. Machine Learning Fundamentals  (quiz-style, conceptual + code)
  2. Data Collection                (pandas, file I/O, manipulation)
  3. Data Processing                (sklearn pipelines, cleaning)
  4. Model Development & Evaluation (training, metrics, validation)

Each function is ready to paste. Scenario triggers are noted inline.
"""

# ══════════════════════════════════════════════════════════════════
# MODULE 1 – MACHINE LEARNING FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════

# ── Linear Regression Prediction ──────────────────────────────────
# Scenario: "Predict y given slope m, intercept b, and input x."

def predict_linear(m: float, b: float, x: float) -> float:
    """y = m*x + b"""
    return m * x + b


# ── L1 vs L2 Regularization ───────────────────────────────────────
# L1 (Lasso)  → drives weights to exactly 0 → automatic feature selection
# L2 (Ridge)  → shrinks weights toward 0 but rarely to exactly 0
# Scenario: "Too many features, some irrelevant?" → use L1
# Scenario: "Multicollinearity? Want all features but smaller weights?" → use L2

def ridge_cost(y_true, y_pred, weights, lambda_=1.0):
    """L2 regularized cost: MSE + lambda * sum(w^2)"""
    import numpy as np
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    return mse + lambda_ * sum(w ** 2 for w in weights)

def lasso_cost(y_true, y_pred, weights, lambda_=1.0):
    """L1 regularized cost: MSE + lambda * sum(|w|)"""
    import numpy as np
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    return mse + lambda_ * sum(abs(w) for w in weights)


# ── Overfitting ────────────────────────────────────────────────────
# Reasons: model too complex, too little data, too many features,
#           training too long (deep nets), no regularization, noisy labels.
# Fixes:   regularization, dropout, more data, cross-validation,
#           early stopping, simpler model, pruning (trees).

def train_test_gap(train_score: float, test_score: float, threshold=0.10) -> str:
    """Quick overfitting diagnostic."""
    gap = train_score - test_score
    if gap > threshold:
        return f"Likely OVERFITTING: gap={gap:.3f} > {threshold}"
    return f"Acceptable gap={gap:.3f}"


# ── Bayes Rule Limitations ────────────────────────────────────────
# 1. Requires accurate prior P(A) – hard in practice.
# 2. Naive Bayes assumes feature independence – often violated.
# 3. Zero-frequency problem: if P(feature|class)=0, posterior=0.
#    → Fix: Laplace (additive) smoothing.
# 4. Sensitive to skewed class priors.

def laplace_smoothing(count, total, n_classes, alpha=1):
    """P(x|class) with Laplace smoothing to avoid zero probabilities."""
    return (count + alpha) / (total + alpha * n_classes)


# ── K in KNN ──────────────────────────────────────────────────────
# Small k  → low bias, high variance (overfitting)
# Large k  → high bias, low variance (underfitting)
# Rule of thumb: k = sqrt(n), always prefer odd k for binary classification.
# Best practice: use cross-validation to tune k.

def choose_k_knn(n_samples: int) -> int:
    """Heuristic starting point for k."""
    import math
    k = int(math.sqrt(n_samples))
    return k if k % 2 == 1 else k + 1   # ensure odd

def knn_cross_val_k(X, y, k_range=range(1, 21)):
    """Find best k via 5-fold cross-validation."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    best_k, best_score = 1, 0
    for k in k_range:
        score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean()
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


# ── GBM vs Random Forest ──────────────────────────────────────────
# Random Forest:  parallel trees, bagging (bootstrap rows + random features)
#                 → robust, hard to overfit, fast to train, good baseline
# GBM:            sequential trees, each corrects predecessor's errors
#                 → higher accuracy ceiling, more hyperparams, slower, can overfit
# Scenario: "Need quick robust baseline?" → Random Forest
# Scenario: "Maximize predictive accuracy, have time to tune?" → GBM / XGBoost

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   random_state=random_state)
    return model.fit(X_train, y_train)

def train_gbm(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = GradientBoostingClassifier(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth)
    return model.fit(X_train, y_train)


# ── Neural Network Fundamentals ───────────────────────────────────
# Key concepts:
#   - Activation functions: ReLU (hidden layers), Sigmoid (binary output),
#     Softmax (multi-class output), Tanh (zero-centered, hidden layers)
#   - Loss: BCE for binary classification, CCE for multi-class, MSE for regression
#   - Backpropagation: chain rule to compute gradients
#   - Vanishing gradient: common with sigmoid/tanh in deep nets → use ReLU
#   - Batch norm, dropout: regularization tricks

import math

def relu(x):          return max(0, x)
def sigmoid(x):       return 1 / (1 + math.exp(-x))
def tanh_(x):         return math.tanh(x)
def softmax(logits):
    e = [math.exp(z - max(logits)) for z in logits]  # numerically stable
    s = sum(e)
    return [v / s for v in e]

def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    total = 0
    for y, p in zip(y_true, y_prob):
        p = max(min(p, 1 - eps), eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(y_true)


# ══════════════════════════════════════════════════════════════════
# MODULE 2 – DATA COLLECTION
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

# ── Filter Even Numbers ───────────────────────────────────────────
def filter_even(numbers: list[int]) -> list[int]:
    return [n for n in numbers if n % 2 == 0]

# ── Filter Names by Starting Letter ──────────────────────────────
def filter_names_by_letter(names: list[str], letter: str) -> list[str]:
    return [n for n in names if n.startswith(letter)]

# ── Read & Combine Files ──────────────────────────────────────────
# Scenario: "Given multiple CSV files, combine them into one DataFrame."

def load_and_combine_csvs(file_paths: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)

def load_and_combine_json(file_paths: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_json(f) for f in file_paths], ignore_index=True)

# ── Core Pandas Operations ────────────────────────────────────────

def explore_dataframe(df: pd.DataFrame) -> dict:
    """Quick EDA summary."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "describe": df.describe().to_dict(),
    }

# Filtering
def filter_rows(df, column, value):
    return df[df[column] == value]

def filter_range(df, column, low, high):
    return df[(df[column] >= low) & (df[column] <= high)]

# Sorting
def sort_df(df, by: list[str], ascending=True):
    return df.sort_values(by=by, ascending=ascending)

# Aggregation
def aggregate(df, group_col, agg_col, func="mean"):
    return df.groupby(group_col)[agg_col].agg(func).reset_index()

def multi_aggregate(df, group_cols, agg_dict):
    """Example agg_dict: {"salary": ["mean","max"], "age": "min"}"""
    return df.groupby(group_cols).agg(agg_dict).reset_index()

# Joining
def join_dataframes(left, right, on, how="inner"):
    return pd.merge(left, right, on=on, how=how)

# GroupBy with transform (keeps original index)
def add_group_mean(df, group_col, value_col, new_col_name):
    df[new_col_name] = df.groupby(group_col)[value_col].transform("mean")
    return df

# Pivot
def pivot(df, index, columns, values, aggfunc="mean"):
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)


# ══════════════════════════════════════════════════════════════════
# MODULE 3 – DATA PROCESSING
# ══════════════════════════════════════════════════════════════════

from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    LabelEncoder, OneHotEncoder,
                                    KBinsDiscretizer)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Average Age (sample question) ────────────────────────────────
def calculate_average_age(people: list[dict]) -> float:
    return round(sum(p["age"] for p in people) / len(people), 2)

# ── Missing Data Handling ─────────────────────────────────────────
def impute_numeric(df, strategy="mean", columns=None):
    """strategy: 'mean', 'median', 'most_frequent', 'constant'"""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    imp = SimpleImputer(strategy=strategy)
    df[cols] = imp.fit_transform(df[cols])
    return df

def impute_categorical(df, strategy="most_frequent", columns=None):
    cols = columns or df.select_dtypes(include="object").columns.tolist()
    imp = SimpleImputer(strategy=strategy)
    df[cols] = imp.fit_transform(df[cols])
    return df

def drop_missing(df, threshold=0.5):
    """Drop columns where more than threshold fraction of values are missing."""
    limit = len(df) * threshold
    return df.dropna(thresh=limit, axis=1)

# ── Scaling ───────────────────────────────────────────────────────
# StandardScaler → mean=0, std=1. Use for: linear models, SVM, NN.
# MinMaxScaler   → [0,1]. Use for: KNN, distance-based models, when distribution not normal.

def scale_standard(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test) if X_test is not None else None
    return X_train_s, X_test_s, scaler

def scale_minmax(X_train, X_test=None):
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test) if X_test is not None else None
    return X_train_s, X_test_s, scaler

# ── Categorical Encoding ──────────────────────────────────────────
# LabelEncoder  → ordinal (ordered) categories only
# OneHotEncoder → nominal (unordered) categories; avoids false ordinal relationships

def label_encode(df, column):
    le = LabelEncoder()
    df[column + "_enc"] = le.fit_transform(df[column])
    return df, le

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

# ── Discretization (Binning) ──────────────────────────────────────
# Scenario: "Convert continuous age into age groups."

def discretize(df, column, n_bins=4, strategy="quantile", encode="ordinal"):
    """strategy: 'uniform', 'quantile', 'kmeans'"""
    kbd = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    df[column + "_bin"] = kbd.fit_transform(df[[column]])
    return df

# ── Variable Transformation ───────────────────────────────────────
def log_transform(df, columns):
    for col in columns:
        df[col + "_log"] = np.log1p(df[col])   # log1p handles 0 safely
    return df

def sqrt_transform(df, columns):
    for col in columns:
        df[col + "_sqrt"] = np.sqrt(df[col])
    return df

# ── Outlier Detection ─────────────────────────────────────────────
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

def flag_outliers_zscore(df, column, threshold=3):
    mean, std = df[column].mean(), df[column].std()
    df[column + "_outlier"] = ((df[column] - mean) / std).abs() > threshold
    return df

# ── Data Leakage ──────────────────────────────────────────────────
# CRITICAL: Never fit scalers/encoders/imputers on the full dataset.
# Always fit on TRAIN only, then transform TRAIN and TEST separately.
# Leakage sources: target-derived features, future data, post-event features.

def safe_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Returns a sklearn Pipeline that prevents leakage.
    Fit on X_train only, then transform both X_train and X_test.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


# ══════════════════════════════════════════════════════════════════
# MODULE 4 – MODEL DEVELOPMENT & EVALUATION
# ══════════════════════════════════════════════════════════════════

from sklearn.model_selection import (train_test_split, cross_val_score,
                                      KFold, LeaveOneOut, GridSearchCV)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_absolute_error, mean_squared_error, r2_score,
                              confusion_matrix, classification_report)

# ── MSE / RMSE (sample question) ─────────────────────────────────
def calculate_mse(predicted: list[float], actual: list[float]) -> float:
    n = len(actual)
    return round(sum((p - a) ** 2 for p, a in zip(predicted, actual)) / n, 2)

def calculate_rmse(predicted: list[float], actual: list[float]) -> float:
    return round(math.sqrt(calculate_mse(predicted, actual)), 2)   # math imported above

# ── Data Splits ───────────────────────────────────────────────────
def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ── Cross-Validation ──────────────────────────────────────────────
def kfold_cv(model, X, y, k=5, scoring="accuracy"):
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=k, shuffle=True, random_state=42),
                             scoring=scoring)
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores.tolist()}

def loocv(model, X, y, scoring="accuracy"):
    """Leave-One-Out CV. Expensive for large datasets."""
    scores = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring=scoring)
    return {"mean": scores.mean(), "std": scores.std()}

# ── Evaluation Metrics: Classification ───────────────────────────
def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    results = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred, average="weighted"),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "report":    classification_report(y_test, y_pred),
    }
    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_prob)
    return results

def gini_coefficient_from_auc(auc: float) -> float:
    """Gini = 2 * AUC - 1"""
    return 2 * auc - 1

# ── Evaluation Metrics: Regression ───────────────────────────────
def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    n, p = len(y_test), X_test.shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return {
        "MAE":         round(mean_absolute_error(y_test, y_pred), 4),
        "MSE":         round(mean_squared_error(y_test, y_pred), 4),
        "RMSE":        round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
        "R2":          round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4),
    }

# ── Hyperparameter Tuning ─────────────────────────────────────────
def grid_search(model, param_grid, X_train, y_train, cv=5, scoring="accuracy"):
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# Example param_grid for RandomForest:
RF_PARAM_GRID = {
    "n_estimators":  [50, 100, 200],
    "max_depth":     [None, 5, 10],
    "min_samples_split": [2, 5],
}

# Example param_grid for GBM:
GBM_PARAM_GRID = {
    "n_estimators":  [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth":     [3, 5, 7],
}

# ── Full End-to-End Template ──────────────────────────────────────
def full_ml_pipeline(df: pd.DataFrame, target_col: str,
                     numeric_features: list, categorical_features: list,
                     model=None):
    """
    Plug-and-play pipeline. Replace model with any sklearn estimator.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = safe_preprocessing_pipeline(numeric_features, categorical_features)

    if model is None:
        model = RandomForestClassifier(random_state=42)

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    val_results  = evaluate_classifier(pipe, X_val, y_val)
    test_results = evaluate_classifier(pipe, X_test, y_test)

    return pipe, val_results, test_results


# ══════════════════════════════════════════════════════════════════
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# ══════════════════════════════════════════════════════════════════
# Scenario: "You have 50 features and suspect many are correlated.
#            Reduce dimensionality before feeding into a model."
# Scenario: "Visualize high-dimensional customer segments in 2D."
# Scenario: "Training is slow / model overfits due to too many features."
#
# Key concepts:
#   - PCA finds directions (principal components) of maximum variance.
#   - Components are orthogonal (uncorrelated) by construction.
#   - ALWAYS scale data before PCA (StandardScaler) – variance is scale-sensitive.
#   - n_components: choose by explained variance ratio (aim for ~95%).
#   - PCA is unsupervised – it ignores the target label y.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def run_pca(X, n_components=None, variance_threshold=0.95, scale=True):
    """
    Full PCA workflow.

    Parameters
    ----------
    X                  : array-like or DataFrame of shape (n_samples, n_features)
    n_components       : int or None. If None, selects components that explain
                         >= variance_threshold of total variance automatically.
    variance_threshold : float (0-1). Used only when n_components=None.
    scale              : bool. If True, StandardScaler is applied first (recommended).

    Returns
    -------
    dict with:
      'X_pca'            – transformed data (n_samples x n_components chosen)
      'explained_ratio'  – variance explained by each component
      'cumulative_ratio' – cumulative explained variance
      'n_components'     – number of components selected
      'loadings'         – how much each original feature contributes to each PC
      'pca'              – fitted PCA object (for transforming new data)
      'scaler'           – fitted scaler (or None)
    """
    # 1. Scale
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 2. Fit full PCA first to inspect explained variance
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    # 3. Auto-select n_components if not specified
    if n_components is None:
        n_components = int(np.argmax(cumulative >= variance_threshold) + 1)

    # 4. Fit final PCA with chosen n_components
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # 5. Loadings: correlation of each original feature with each PC
    feature_names = (list(X.columns) if hasattr(X, "columns")
                     else [f"feature_{i}" for i in range(pca.components_.shape[1])])
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return {
        "X_pca":           X_pca,
        "explained_ratio": pca.explained_variance_ratio_,
        "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
        "n_components":    n_components,
        "loadings":        loadings,
        "pca":             pca,
        "scaler":          scaler,
    }


def pca_summary(result: dict) -> None:
    """Print a human-readable PCA summary."""
    print(f"Components selected : {result['n_components']}")
    print(f"{'PC':<6} {'Explained':>10} {'Cumulative':>12}")
    print("-" * 30)
    for i, (ev, cum) in enumerate(zip(result["explained_ratio"],
                                      result["cumulative_ratio"]), 1):
        print(f"PC{i:<4} {ev:>9.1%} {cum:>11.1%}")
    print("\nTop feature loadings per component:")
    print(result["loadings"].abs()
                            .apply(lambda col: col.nlargest(3).index.tolist())
                            .to_string())


def pca_transform_new(result: dict, X_new):
    """
    Apply fitted PCA (and scaler) to new/test data.
    Always use this instead of re-fitting on test data.
    """
    X_new = result["scaler"].transform(X_new) if result["scaler"] else X_new
    return result["pca"].transform(X_new)


# ── Quick Usage ────────────────────────────────────────────────────
# result = run_pca(X_train)               # auto-selects components for 95% variance
# pca_summary(result)                     # print explained variance table
# X_train_pca = result["X_pca"]          # use this as features for your model
# X_test_pca  = pca_transform_new(result, X_test)   # transform test set safely
#
# Visualize in 2D:
# result_2d = run_pca(X, n_components=2)
# import matplotlib.pyplot as plt
# plt.scatter(result_2d["X_pca"][:,0], result_2d["X_pca"][:,1], c=y)
# plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA 2D projection")
# plt.show()


# ══════════════════════════════════════════════════════════════════
# QUICK-REFERENCE DECISION TABLE
# ══════════════════════════════════════════════════════════════════
"""
SCENARIO                                         │ TOOL / FUNCTION
─────────────────────────────────────────────────┼──────────────────────────────────────
Predict continuous value from features           │ LinearRegression → predict_linear / linear_regression_ols
Binary classification (churn, click)             │ LogisticRegression / RandomForest / GBM
Multiclass classification                        │ RandomForest + softmax output
Too many irrelevant features                     │ L1 (Lasso) regularization
Multicollinearity / all features useful          │ L2 (Ridge) regularization
Model memorizes training data                    │ Overfitting → regularize, dropout, more data
Choose k for KNN                                 │ choose_k_knn() → cross-validate with knn_cross_val_k()
Fast robust baseline                             │ Random Forest (train_random_forest)
Best accuracy, time to tune                      │ GBM / XGBoost (train_gbm + grid_search)
P(disease | positive test)                       │ bayes() – watch for low prevalence!
Avoid zero-probability in Naive Bayes            │ laplace_smoothing()
Combine multiple CSV/JSON files                  │ load_and_combine_csvs / load_and_combine_json
Fill missing numeric values                      │ impute_numeric(strategy="median")
Fill missing categorical values                  │ impute_categorical(strategy="most_frequent")
Normalize for distance-based models (KNN)        │ scale_minmax()
Normalize for linear/neural models               │ scale_standard()
Unordered categories (color, region)             │ one_hot_encode()
Ordered categories (low/mid/high)                │ label_encode()
Skewed numeric feature                           │ log_transform() or sqrt_transform()
Outlier removal                                  │ remove_outliers_iqr() (robust) / flag_outliers_zscore()
Prevent data leakage                             │ safe_preprocessing_pipeline() – fit on train ONLY
Evaluate classification model                    │ evaluate_classifier() → accuracy, F1, AUC, Gini
Evaluate regression model                        │ evaluate_regressor() → MAE, RMSE, R², Adj-R²
Validate with limited data                       │ kfold_cv(k=5) or loocv()
Tune hyperparameters                             │ grid_search(model, param_grid, ...)
Too many correlated features / slow training     │ run_pca() → auto-selects components for 95% variance
Visualize clusters or segments in 2D            │ run_pca(X, n_components=2) → scatter plot PC1 vs PC2
Apply PCA to test set without leakage           │ pca_transform_new(result, X_test)
"""