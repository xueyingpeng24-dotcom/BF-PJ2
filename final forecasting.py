# %%
# =============================================================================
# Import Required Libraries
# =============================================================================

# Standard Library
import json
import warnings
from pathlib import Path


# Numerical Computing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
# seaborn is used for statistical visualisations (e.g., correlation heatmap).
import seaborn as sns

# Ignore warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Forecasting Libraries (Nixtla)
# =============================================================================
# MLForecast is used to build the global forecasting model with
# automatic time-series feature engineering (lags, rolling statistics,
# and calendar features), following the framework introduced in the course.

from statsforecast import StatsForecast
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean


#  pmdarima is used to implement the AutoARIMA benchmark model.
try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None

# =============================================================================
# Machine Learning
# =============================================================================

from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# Evaluation Metrics
# =============================================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

# Notebook compatibility:
# Import display explicitly so that display() remains
# available when the notebook is converted to a Python (.py) file.
from IPython.display import display



# Project directories

# Define the project root directory and locate the raw and
# processed data folders. If the script is executed from a
# subdirectory, move one level up to ensure the correct
# project structure is used.

project_dir = Path.cwd()

if not (project_dir / "data").exists():
    project_dir = project_dir.parent

raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"

# Create the processed data directory if it does not already exist.
processed_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Exploratory Data Analysis (EDA)
# =============================================================================
# This section explores the sales, future, and metadata datasets to
# understand their structure, assess data quality, identify temporal
# patterns and business drivers, and provide insights for subsequent
# feature engineering and forecasting model development.
# =============================================================================


# ================================================================
# Display Settings (EDA Only)
# =============================================================================
# Adjust pandas display options to ensure that descriptive statistics
# and DataFrames are shown completely during exploratory data analysis.
# These settings only affect the display and do not modify the data.
# Display all columns without truncation during exploratory analysis.
pd.set_option("display.max_columns", None)

# Remove the display width limitation for console output.
pd.set_option("display.width", None)

# Display full column contents without truncation.
pd.set_option("display.max_colwidth", None)


# =============================================================================
# STEP 2: Read the Dataset and Ensure Store ID Consistency
# =============================================================================

# Resolve the data directory robustly before reading any files.
# This avoids NameError when the notebook is executed from a different
# working directory and keeps the path logic consistent across runs.

raw_dir = Path.cwd() / "data" / "raw"
if not raw_dir.exists():
    raw_dir = Path.cwd().parent / "data" / "raw"

sales_data_path = raw_dir / "sales_data.csv"
future_data_path = raw_dir / "future_values.csv"
meta_data_path = raw_dir / "metadata.csv"

for path in [sales_data_path, future_data_path, meta_data_path]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# Read the sales dataset.
# low_memory=False prevents pandas from inferring column types in chunks,
# which avoids unnecessary DtypeWarning messages for larger datasets.

sales_data = pd.read_csv(
    sales_data_path,
    low_memory=False
)

future_data = pd.read_csv(
    future_data_path,
    low_memory=False
)

meta_data = pd.read_csv(
    meta_data_path,
    low_memory=False
)

# Compute store id sets after the CSVs have been read to avoid NameError
sales_store = set(sales_data["store_id"])
future_store = set(future_data["store_id"])
meta_store = set(meta_data["store_id"])

print("Sales stores :", len(sales_store))
print("Future stores:", len(future_store))
print("Meta stores  :", len(meta_store))

print("\nSales == Future :", sales_store == future_store)
print("Sales == Meta   :", sales_store == meta_store)
print("Future == Meta  :", future_store == meta_store)

# We have same store id in all files


# =============================================================================
# STEP 3: Inspect Dataset Dimensions
# =============================================================================
# Understanding the size of the dataset is an important first step.
# The number of rows indicates the total number of observations,
# while the number of columns indicates the number of available variables.

print("=" * 60)
print("DATASET DIMENSIONS")
print("=" * 60)
print(sales_data.shape)


# =============================================================================
# STEP 4: Inspect Dataset Structure
# =============================================================================
# info() provides a concise overview of the dataset, including:
# - column names
# - data types
# - number of non-missing observations
#
# This helps identify potential missing values and variables that require
# type conversion before further analysis.

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)
sales_data.info()


# =============================================================================
# STEP 5: Inspect Data Types
# =============================================================================
# Display the data type of each variable.
# This is useful to verify whether variables have been imported correctly.
# For example, the date column is typically read as an object and should later
# be converted into pandas datetime format.

print("\n" + "=" * 60)
print("VARIABLE DATA TYPES")
print("=" * 60)
print(sales_data.dtypes)

# =============================================================================
# STEP 6: Display Sample Observations
# =============================================================================
# Displaying the first few rows provides a quick visual inspection of the
# dataset and helps verify that the variables contain reasonable values.

print("\n" + "=" * 60)
print("FIRST FIVE OBSERVATIONS")
print("=" * 60)
print(sales_data.head())


# =============================================================================
# STEP 7: Convert the Date Variable
# =============================================================================
# The date column is currently stored as an object (string).
# Converting it to datetime enables chronological sorting,
# time-based aggregation and extraction of calendar features
# (e.g. weekday, month and week), which are essential for time series analysis.

sales_data["date"] = pd.to_datetime(sales_data["date"])


# =============================================================================
# STEP 8: Verify Date Range and Continuity Check
# =============================================================================
# Inspect the temporal coverage of the dataset.
# Knowing the start and end dates provides an overview of the available
# historical period and helps verify that the dataset has been loaded correctly.
# Ensure that there is no gaps in time series

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)
print("Start Date:", sales_data["date"].min())
print("End Date  :", sales_data["date"].max())
print("Unique Dates:", sales_data["date"].nunique())

print("\n" + "=" * 60)
print("DATE CONTINUITY CHECK")
print("=" * 60)

stores_with_gaps = {}

for store, group in sales_data.groupby("store_id"):

    dates = group["date"].sort_values()

    expected = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq="D"
    )

    missing = expected.difference(dates)

    if len(missing) > 0:

        stores_with_gaps[store] = len(missing)

print("Stores checked:",
      sales_data["store_id"].nunique())

print("Stores with missing dates:",
      len(stores_with_gaps))

if len(stores_with_gaps) == 0:

    print("All stores have complete daily timelines.")

else:

    print("\nFirst 10 stores with gaps:")

    for store, n in list(stores_with_gaps.items())[:10]:

        print(f"{store}: {n} missing days")

print("Start:", sales_data["date"].min())
print("End:  ", sales_data["date"].max())

sales_data["date"] = pd.to_datetime(sales_data["date"])

results = []

for store, group in sales_data.groupby("store_id"):
    actual_dates = pd.DatetimeIndex(group["date"].sort_values().unique())

    expected_dates = pd.date_range(
        start=actual_dates.min(),
        end=actual_dates.max(),
        freq="D"
    )

    missing_dates = expected_dates.difference(actual_dates)

    results.append({
        "store_id": store,
        "start_date": actual_dates.min(),
        "end_date": actual_dates.max(),
        "actual_days": len(actual_dates),
        "expected_days": len(expected_dates),
        "missing_days": len(missing_dates)
    })

date_check = pd.DataFrame(results)

display(date_check.head())

summary = (
    date_check["missing_days"]
    .value_counts()
    .sort_index()
    .reset_index()
)

summary.columns = ["missing_days", "number_of_stores"]

display(summary)

# =============================================================================
# STEP 9: Check Missing Values
# =============================================================================
# Missing values may negatively affect both exploratory analysis and model
# training. Therefore, the number of missing observations is calculated
# for every variable before any preprocessing is performed.

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(sales_data.isnull().sum())


# =============================================================================
# STEP 10: Check Duplicate Observations
# =============================================================================
# Duplicate records can introduce bias into model estimation.
# Both complete duplicates and duplicate store-date combinations should be
# investigated.

print("\n" + "=" * 60)
print("DUPLICATE OBSERVATIONS")
print("=" * 60)
print("Entire duplicated rows:",
      sales_data.duplicated().sum())

print("Duplicate store-date pairs:",
      sales_data.duplicated(
          subset=["store_id", "date"]
      ).sum())


# =============================================================================
# STEP 11: Descriptive Statistics
# =============================================================================
# Descriptive statistics are reported separately for numerical,
# categorical and datetime variables.
#
# Splitting the summaries by data type improves readability and provides
# statistics that are meaningful for each variable type.

# -------------------------------------------------------------------------
# Numerical Variables
# -------------------------------------------------------------------------
# Summary statistics for numerical variables provide information about
# central tendency, variability and potential extreme values.

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)
print(sales_data.describe(include="number"))


# -------------------------------------------------------------------------
# Categorical Variables
# -------------------------------------------------------------------------
# Summary statistics for categorical variables provide information about
# the number of unique categories, the most frequent category and its
# occurrence frequency.

print("\n" + "=" * 60)
print("CATEGORICAL SUMMARY")
print("=" * 60)
print(sales_data.describe(include="object"))


# =============================================================================
# STEP 12: Inspect Categorical Variables
# =============================================================================
# Display the distribution of categorical variables.
# Understanding category frequencies is useful for later feature engineering
# and evaluating whether some categories occur very infrequently.

print("\n" + "=" * 60)
print("STATE HOLIDAY DISTRIBUTION")
print("=" * 60)
print(sales_data["state_holiday"].value_counts(dropna=False))

print("\nOPEN STATUS (Count)")
print(sales_data["open"].value_counts())

print("\nOPEN STATUS (Percentage)")
print(
    sales_data["open"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPROMOTION STATUS")
print(sales_data["promo"].value_counts())
print(sales_data["promo"].value_counts(normalize=True))

print("\nSCHOOL HOLIDAY STATUS")
print(sales_data["school_holiday"].value_counts())
print(sales_data["school_holiday"].value_counts(normalize=True))



# =============================================================================
# STEP 13: Relationship Between Store Status and Sales
# =============================================================================
# Verify whether closed stores (open = 0) consistently report zero sales
# and zero customers. This validation helps identify whether closed-store
# observations should be treated differently during later modelling.

print("\n" + "=" * 60)
print("STORE STATUS VS SALES")
print("=" * 60)

print(
    sales_data.groupby("open")[["sales", "customers"]]
    .agg(["min", "max", "mean"])
)

closed_store = sales_data[sales_data["open"] == 0]

print(
    "\nClosed stores with non-zero sales:",
    (closed_store["sales"] != 0).sum()
)

print(
    "Closed stores with non-zero customers:",
    (closed_store["customers"] != 0).sum()
)

#sales data  EDA

# =============================================================================
# STEP 16: Sales Distribution
# =============================================================================
# Visualise the distribution of daily sales to understand
# its overall shape, spread and potential skewness.

plt.figure(figsize=(8,5))

plt.hist(
    sales_data["sales"],
    bins=50
)

plt.title("Distribution of Daily Sales")
plt.xlabel("Daily Sales")
plt.ylabel("Frequency")

plt.savefig(
    "../results/figures/sales_distribution.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8,2))

plt.boxplot(
    sales_data["sales"],
    vert=False
)

plt.title("Boxplot of Daily Sales")
plt.xlabel("Daily Sales")

plt.savefig(
    "../results/figures/sales_boxplot.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# =============================================================================
# Correlation Analysis
# =============================================================================

# Merge competition distance for correlation analysis
corr_data = sales_data.merge(
    meta_data[["store_id", "competition_distance"]],
    on="store_id",
    how="left"
)

# Select numerical variables
corr_columns = [
    "sales",
    "customers",
    "promo",
    "school_holiday",
    "competition_distance"
]
correlation = corr_data[corr_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

# ----------------------------------------------------------
# Save Figure
# ----------------------------------------------------------


plt.savefig(
    "../results/figures/correlation_matrix.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print("\n" + "=" * 60)
print("CORRELATION WITH SALES")
print("=" * 60)

print(
    correlation["sales"]
    .sort_values(ascending=False)
)
# =============================================================================
# STEP 17: Daily Total Sales Trend
# =============================================================================
# Aggregate sales across all stores to examine
# the overall sales trend over time.

daily_sales = (
    sales_data
    .groupby("date", as_index=False)["sales"]
    .sum()
)
plt.figure(figsize=(12,5))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    linewidth=1
)

plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")

plt.tight_layout()
plt.show()

#The aggregated daily sales exhibit a clear weekly seasonal pattern, with regular drops corresponding to store closures. No strong long-term upward or downward trend is observed over the study period, although several temporary sales peaks occur, likely associated with promotional campaigns or holiday periods.

# =============================================================================
# STEP 17.2: Daily Sales Trend with 7-Day Rolling Mean
# =============================================================================
# Apply a 7-day rolling average to smooth daily fluctuations
# and highlight the underlying sales trend.

daily_sales["rolling_7"] = (
    daily_sales["sales"]
    .rolling(window=7)
    .mean()
)

plt.figure(figsize=(12,5))

# Original daily sales
plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    color="lightgray",
    linewidth=0.8,
    label="Daily Sales"
)

# 7-day rolling mean
plt.plot(
    daily_sales["date"],
    daily_sales["rolling_7"],
    color="red",
    linewidth=2,
    label="7-Day Rolling Mean"
)

plt.title("Daily Sales with 7-Day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")

plt.legend()

plt.tight_layout()

plt.show()
# =============================================================================
# STEP 18: Weekly and Month Sales Pattern
# =============================================================================
# Examine average sales by day of week and month to identify
# weekly seasonality.

sales_data["weekday"] = sales_data["date"].dt.day_name()

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_pattern = (
    sales_data
    .groupby("weekday")["sales"]
    .mean()
    .reindex(weekday_order)
)
plt.figure(figsize=(9,5))

plt.bar(
    weekly_pattern.index,
    weekly_pattern.values
)

plt.title("Average Daily Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.show()

# =============================================================================
# STEP 18.2: Weekly Sales Pattern (Open Stores Only)
# =============================================================================
# Analyse weekly sales pattern after excluding closed stores.

open_sales = sales_data[sales_data["open"] == 1].copy()
open_sales["weekday"] = open_sales["date"].dt.day_name()

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_pattern_open = (
    open_sales
    .groupby("weekday")["sales"]
    .mean()
    .reindex(weekday_order)
)
plt.figure(figsize=(9,5))

plt.bar(
    weekly_pattern_open.index,
    weekly_pattern_open.values
)

plt.title("Average Daily Sales by Day of Week (Open Stores Only)")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.show()


# =============================================================================
# STEP 18.3: Monthly Sales Pattern
# =============================================================================
# Examine average sales by month to identify
# monthly seasonality.

if "open_sales" not in globals():
    open_sales = sales_data[sales_data["open"] == 1].copy()

sales_data["month"] = sales_data["date"].dt.month_name()

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_pattern = (
    sales_data
    .groupby("month")["sales"]
    .mean()
    .reindex(month_order)
)
plt.figure(figsize=(9,5))

plt.bar(
    monthly_pattern.index,
    monthly_pattern.values
)

plt.title("Average Daily Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.show()

# =============================================================================
# STEP 18.4: Monthly Sales Pattern (Open Stores Only)
# =============================================================================
# Analyse monthly sales pattern after excluding closed stores.

open_sales["month"] = open_sales["date"].dt.month_name()

monthly_pattern_open = (
    open_sales
    .groupby("month")["sales"]
    .mean()
    .reindex(month_order)
)
plt.figure(figsize=(9,5))

plt.bar(
    monthly_pattern_open.index,
    monthly_pattern_open.values
)

plt.title("Average Daily Sales by Month (Open Stores Only)")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.show()

# =============================================================================
# STEP 19: Promotion Effect
# =============================================================================
# Compare average sales between promotional and non-promotional days.

promo_summary = (
    sales_data
    .groupby("promo")["sales"]
    .agg(
        mean_sales="mean",
        median_sales="median",
        std_sales="std",
        count="count"
    )
)

print("\n" + "=" * 60)
print("PROMOTION EFFECT")
print("=" * 60)

print(promo_summary)
plt.figure(figsize=(6,5))

promo_mean = (
    sales_data
    .groupby("promo")["sales"]
    .mean()
)

plt.bar(
    ["No Promotion", "Promotion"],
    promo_mean.values
)

plt.title("Average Sales by Promotion Status")
plt.xlabel("Promotion")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.show()

open_sales = sales_data[
    sales_data["open"] == 1
]
promo_mean_open = (
    open_sales
    .groupby("promo")["sales"]
    .mean()
)

plt.figure(figsize=(6,5))

plt.bar(
    ["No Promotion", "Promotion"],
    promo_mean_open.values
)

plt.title("Average Sales by Promotion Status (Open Stores Only)")
plt.xlabel("Promotion")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.show()


# =============================================================================
# STEP 20: State Holiday Effect
# =============================================================================
# Compare average sales across different state holiday types.

holiday_summary = (
    sales_data
    .groupby("state_holiday")["sales"]
    .agg(
        mean_sales="mean",
        median_sales="median",
        std_sales="std",
        count="count"
    )
)

print("\n" + "=" * 60)
print("STATE HOLIDAY EFFECT")
print("=" * 60)

print(holiday_summary)

holiday_mean = (
    sales_data
    .groupby("state_holiday")["sales"]
    .mean()
)

plt.figure(figsize=(7,5))

plt.bar(
    holiday_mean.index,
    holiday_mean.values
)

plt.title("Average Sales by State Holiday")
plt.xlabel("State Holiday")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.show()

holiday_mean_open = (
    open_sales
    .groupby("state_holiday")["sales"]
    .mean()
)

plt.figure(figsize=(7,5))

plt.bar(
    holiday_mean_open.index,
    holiday_mean_open.values
)

plt.title("Average Sales by State Holiday (Open Stores Only)")
plt.xlabel("State Holiday")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.show()


#When all observations are included, average sales on state holidays appear substantially lower than on normal trading days.
#However, after excluding closed-store observations, sales during state holidays exceed those on normal trading days. This suggests that the apparent decline in holiday sales is primarily due to widespread store closures rather than reduced customer demand.
# =============================================================================
# STEP 21: Relationship Between Customers and Sales
# =============================================================================
# Explore the relationship between customer numbers and sales.

sample_data = sales_data.sample(
    n=10000,
    random_state=42
)

plt.figure(figsize=(7,6))

plt.scatter(
    sample_data["customers"],
    sample_data["sales"],
    alpha=0.3,
    s=10
)

plt.title("Sales vs Customers")
plt.xlabel("Number of Customers")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()

correlation = sales_data["customers"].corr(
    sales_data["sales"]
)

print("\n" + "=" * 60)
print("CUSTOMERS AND SALES CORRELATION")
print("=" * 60)

print(f"Pearson correlation: {correlation:.4f}")
# future data  inspection
# =============================================================================
# STEP 1: Dataset Overview
# =============================================================================

print("\n" + "=" * 60)
print("DATASET DIMENSIONS")
print("=" * 60)
print(future_data.shape)

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)
future_data.info()

print("\n" + "=" * 60)
print("VARIABLE DATA TYPES")
print("=" * 60)
print(future_data.dtypes)

print("\n" + "=" * 60)
print("FIRST FIVE OBSERVATIONS")
print("=" * 60)
print(future_data.head())

# =============================================================================
# STEP 2: Date and Data Quality
# =============================================================================

future_data["date"] = pd.to_datetime(future_data["date"])

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)

print("Start Date :", future_data["date"].min())
print("End Date   :", future_data["date"].max())
print("Unique Dates:", future_data["date"].nunique())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(future_data.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE OBSERVATIONS")
print("=" * 60)

print(
    "Entire duplicated rows:",
    future_data.duplicated().sum()
)

print(
    "Duplicate store-date pairs:",
    future_data.duplicated(
        subset=["store_id", "date"]
    ).sum()
)

# =============================================================================
# STEP 3: Categorical Variables
# =============================================================================

print("\n" + "=" * 60)
print("OPEN STATUS")
print("=" * 60)

print(future_data["open"].value_counts())
print(future_data["open"].value_counts(normalize=True))


print("\nPROMOTION STATUS")

print(future_data["promo"].value_counts())
print(future_data["promo"].value_counts(normalize=True))

print("\nSTATE HOLIDAY")

print(future_data["state_holiday"].value_counts())

print("\nSCHOOL HOLIDAY")

print(future_data["school_holiday"].value_counts())
print(future_data["school_holiday"].value_counts(normalize=True))

# =============================================================================
# STEP 4: Store-Level Observations
# =============================================================================

store_counts = (
    future_data
    .groupby("store_id")["date"]
    .count()
)

print("\n" + "=" * 60)
print("STORE-LEVEL OBSERVATIONS")
print("=" * 60)

print(store_counts.describe())

print("\nNumber of stores:",
      store_counts.shape[0])

print("Stores with complete history:",
      (store_counts == store_counts.max()).sum())

# =============================================================================
# STEP 5: Customers Availability
# =============================================================================

print("\n" + "=" * 60)
print("CUSTOMERS VARIABLE")
print("=" * 60)

print("Missing values:",
      future_data["customers"].isnull().sum())

print("Total records:",
      len(future_data))
# =============================================================================
# STEP 6: Missing Open Status
# =============================================================================
# Display records where the store open status is missing.

print("\n" + "=" * 60)
print("MISSING OPEN STATUS")
print("=" * 60)

print(
    future_data[
        future_data["open"].isna()
    ]
)
store379_history = sales_data[
    sales_data["store_id"] == "store_379"
].copy()

store379_history["date"] = pd.to_datetime(store379_history["date"])

store379_history = store379_history.sort_values("date")

print(
    store379_history.tail(30)[
        ["date", "open", "sales", "promo"]
    ]
)

# =============================================================================
# METADATA INITIAL DATA INSPECTION
# =============================================================================
# STEP 1: Dataset Dimensions
# =============================================================================

print("\n" + "=" * 60)
print("DATASET DIMENSIONS")
print("=" * 60)

print(meta_data.shape)

# =============================================================================
# STEP 2: Dataset Structure
# =============================================================================

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

meta_data.info()

print("\n" + "=" * 60)
print("VARIABLE DATA TYPES")
print("=" * 60)

print(meta_data.dtypes)

# =============================================================================
# STEP 3: First Five Observations
# =============================================================================

print("\n" + "=" * 60)
print("FIRST FIVE OBSERVATIONS")
print("=" * 60)

print(meta_data.head())

# =============================================================================
# STEP 4: Missing Values
# =============================================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(meta_data.isnull().sum())

# =============================================================================
# STEP 5: Duplicate Observations
# =============================================================================

print("\n" + "=" * 60)
print("DUPLICATE OBSERVATIONS")
print("=" * 60)

print("Entire duplicated rows:",
      meta_data.duplicated().sum())

print("Duplicate store_id:",
      meta_data["store_id"].duplicated().sum())

# =============================================================================
# STEP 6: Numerical Summary
# =============================================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(meta_data.describe())

# =============================================================================
# STEP 7: Categorical Summary
# =============================================================================

print("\n" + "=" * 60)
print("STORE TYPE DISTRIBUTION")
print("=" * 60)

print(meta_data["store_type"].value_counts())

print("\nPercentage:")

print(
    meta_data["store_type"]
    .value_counts(normalize=True)
    .round(4)
)

print("\n" + "=" * 60)
print("ASSORTMENT DISTRIBUTION")
print("=" * 60)

print(meta_data["assortment"].value_counts())

print("\nPercentage:")

print(
    meta_data["assortment"]
    .value_counts(normalize=True)
    .round(4)
)

# =============================================================================
# STEP 8: Competition Distance
# =============================================================================

print("\n" + "=" * 60)
print("COMPETITION DISTANCE")
print("=" * 60)

print(meta_data["competition_distance"].describe())

print("\nMinimum distance:",
      meta_data["competition_distance"].min())

print("Maximum distance:",
      meta_data["competition_distance"].max())

# =============================================================================
# STEP 9: Store Coverage
# =============================================================================

sales_stores = set(sales_data["store_id"])
future_stores = set(future_data["store_id"])
meta_stores = set(meta_data["store_id"])

print("\n" + "=" * 60)
print("STORE COVERAGE CHECK")
print("=" * 60)

print("Stores in sales_data :", len(sales_stores))
print("Stores in future_data:", len(future_stores))
print("Stores in metadata   :", len(meta_stores))

print("\nStores only in sales_data:")
print(sales_stores - meta_stores)

print("\nStores only in future_data:")
print(future_stores - meta_stores)

print("\nStores only in metadata:")
print(meta_stores - sales_stores)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("METADATA INSPECTION COMPLETED")
print("=" * 60)

print("Total stores :", len(meta_stores))
print("Missing values:")
print(meta_data.isnull().sum().sum())
print("Duplicate rows:", meta_data.duplicated().sum())




# =============================================================================
# Data Cleaning and Preparation
# =============================================================================
# This section performs data cleaning and preprocessing to ensure data
# consistency and quality before modelling. It includes data type
# conversion, missing value treatment, metadata integration, validation,
# and the creation of processed datasets for subsequent forecasting.
# =============================================================================
# STEP 1: Data Type Conversion
# =============================================================================

# Convert date columns
sales_data["date"] = pd.to_datetime(sales_data["date"])
future_data["date"] = pd.to_datetime(future_data["date"])

# Convert categorical variables in sales_data
sales_data[
    ["store_id", "state_holiday"]
] = sales_data[
    ["store_id", "state_holiday"]
].astype("category")

# Convert categorical variables in future_data
future_data[
    ["store_id", "state_holiday"]
] = future_data[
    ["store_id", "state_holiday"]
].astype("category")

# Convert categorical variables in metadata
meta_data[
    ["store_id", "store_type", "assortment"]
] = meta_data[
    ["store_id", "store_type", "assortment"]
].astype("category")

# =============================================================================
# STEP 2: Missing Value Summary
# =============================================================================

print("=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)

print("\nSales Data")
print(sales_data.isnull().sum())

print("\nFuture Data")
print(future_data.isnull().sum())

print("\nMetadata")
print(meta_data.isnull().sum())

# Fill missing values in the 'open' column.
# The missing values are limited to 11 consecutive records of store_379.
# Based on the historical operating pattern of this store,
# the store is normally open on weekdays and Saturdays.
# Therefore, the missing values are imputed as 1 (open).

future_data.loc[
    (future_data["store_id"] == "store_379") &
    (future_data["open"].isna()),
    "open"
] = 1

# Fill the missing competition distance using the median.
# Only one value is missing and the variable is positively skewed,
# therefore the median provides a robust estimate.

median_distance = meta_data["competition_distance"].median()

meta_data["competition_distance"] = (
    meta_data["competition_distance"]
    .fillna(median_distance)
)

# The 'customers' column in future_data is intentionally left unchanged.
# Customer numbers are unavailable for the forecasting horizon and
# will not be used as an input feature during model development.

# =============================================================================
# STEP 3: Validation
# =============================================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print("\nSales Data")
print(sales_data.isnull().sum())

print("\nFuture Data")
print(future_data.isnull().sum())

print("\nMetadata")
print(meta_data.isnull().sum())
# =============================================================================
# STEP 4: Merge Metadata
# =============================================================================
# Merge metadata into the historical sales data.

sales_clean = sales_data.merge(
    meta_data,
    on="store_id",
    how="left"
)
# Merge metadata into the future dataset.

future_clean = future_data.merge(
    meta_data,
    on="store_id",
    how="left"
)

# =============================================================================
# STEP 5: Merge Validation
# =============================================================================

print("\n" + "=" * 60)
print("DATASET SHAPES AFTER MERGE")
print("=" * 60)

print("Sales before merge :", sales_data.shape)
print("Sales after merge  :", sales_clean.shape)

print()

print("Future before merge:", future_data.shape)
print("Future after merge :", future_clean.shape)

print("\n" + "=" * 60)
print("MISSING VALUES AFTER MERGE")
print("=" * 60)

print("\nSales Clean")
print(sales_clean.isnull().sum())

print("\nFuture Clean")
print(future_clean.isnull().sum())

print("\n" + "=" * 60)
print("NEW VARIABLES")
print("=" * 60)

print(sales_clean.columns)
from pathlib import Path
import pandas as pd

raw_dir = Path.cwd() / "data" / "raw"
if not raw_dir.exists():
    raw_dir = Path.cwd().parent / "data" / "raw"

sales_data_path = raw_dir / "sales_data.csv"
future_data_path = raw_dir / "future_values.csv"
meta_data_path = raw_dir / "metadata.csv"

sales_data = pd.read_csv(sales_data_path, low_memory=False)
future_data = pd.read_csv(future_data_path, low_memory=False)
meta_data = pd.read_csv(meta_data_path, low_memory=False)

sales_data["date"] = pd.to_datetime(sales_data["date"])
future_data["date"] = pd.to_datetime(future_data["date"])

future_data.loc[
    (future_data["store_id"] == "store_379") &
    (future_data["open"].isna()),
    "open"
] = 1

median_distance = meta_data["competition_distance"].median()
meta_data["competition_distance"] = meta_data["competition_distance"].fillna(median_distance)

sales_clean = sales_data.merge(meta_data, on="store_id", how="left")
future_clean = future_data.merge(meta_data, on="store_id", how="left")

output_dir = Path(".")
if output_dir.name != "notebooks":
    output_dir = output_dir / "notebooks"
output_dir.mkdir(parents=True, exist_ok=True)

sales_clean.to_csv(output_dir / "sales-meta.csv", index=False)
future_clean.to_csv(output_dir / "future-meta.csv", index=False)


print("Saved:")
print(output_dir / "future-meta.csv")
print(output_dir / "sales-meta.csv")

# Load and prepare data
raw_dir = Path.cwd() / "data" / "raw"
if not raw_dir.exists():
    raw_dir = Path.cwd().parent / "data" / "raw"

sales_data = pd.read_csv(raw_dir / "sales_data.csv", low_memory=False)
future_data = pd.read_csv(raw_dir / "future_values.csv", low_memory=False)
meta_data = pd.read_csv(raw_dir / "metadata.csv", low_memory=False)

sales_data["date"] = pd.to_datetime(sales_data["date"])
future_data["date"] = pd.to_datetime(future_data["date"])

future_data.loc[
    (future_data["store_id"] == "store_379") &
    (future_data["open"].isna()),
    "open"
] = 1

median_distance = meta_data["competition_distance"].median()
meta_data["competition_distance"] = meta_data["competition_distance"].fillna(median_distance)

sales_clean = sales_data.merge(meta_data, on="store_id", how="left")
future_clean = future_data.merge(meta_data, on="store_id", how="left")

# Create processed folder if it does not exist
Path("../data/processed").mkdir(parents=True, exist_ok=True)

sales_clean.to_csv(
    "../data/processed/sales_clean.csv",
    index=False
)

future_clean.to_csv(
    "../data/processed/future_clean.csv",
    index=False
)

print("Processed datasets saved.")



#%%
## Benchmark Forecasting

# We evaluate benchmark models separately for each store.
# Each store uses its own history to create 42-day forecasts, and the metrics are stored per store and per model.

# This benchmark evaluates several classical forecasting methods
# (Mean, Naive, Drift, Seasonal Naive and AutoARIMA) separately
# for each individual store.
#
# Runtime Notice:
# On our development machine, this benchmark requires
# approximately 20 hours to complete.
#
# The long runtime is mainly caused by AutoARIMA, which performs
# automatic model selection for every store independently.
# For each store, AutoARIMA searches multiple ARIMA and Seasonal
# ARIMA parameter combinations before selecting the final model.
#
# Since the dataset contains several hundred stores, this model
# selection procedure is repeated hundreds of times, making
# AutoARIMA by far the most computationally expensive benchmark.
#
# This benchmark is included primarily for model comparison rather
# than the final forecasting framework.
#
# ==========================================================
import warnings
warnings.filterwarnings("ignore")


# ==========================================================
# Load data
# ==========================================================

sales = pd.read_csv(processed_dir / "sales_clean.csv")
sales["date"] = pd.to_datetime(sales["date"])
sales = sales.sort_values(["store_id", "date"])

# ==========================================================
# Helper functions
# ==========================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_forecasts(train_series, test_len):
    forecasts = {}

    train_series = pd.Series(train_series).astype(float)
    train_series = train_series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(train_series) < 10:
        return {
            "Mean": np.repeat(np.nan, test_len),
            "Naive": np.repeat(np.nan, test_len),
            "Drift": np.repeat(np.nan, test_len),
            "Seasonal Naive": np.repeat(np.nan, test_len),
            "AutoARIMA": np.repeat(np.nan, test_len),
        }

    forecasts["Mean"] = np.full(test_len, train_series.mean())
    forecasts["Naive"] = np.full(test_len, train_series.iloc[-1])

    drift = np.arange(1, test_len + 1)
    forecasts["Drift"] = train_series.iloc[-1] + drift * (
        (train_series.iloc[-1] - train_series.iloc[0]) / (len(train_series) - 1)
    )

    forecasts["Seasonal Naive"] = np.array([
        train_series.iloc[-7 + (i % 7)] for i in range(test_len)
    ])

    if auto_arima is not None:
        try:
            model = auto_arima(
                train_series,
                seasonal=True,
                m=7,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            forecasts["AutoARIMA"] = model.predict(n_periods=test_len)
        except Exception:
            forecasts["AutoARIMA"] = np.repeat(np.nan, test_len)
    else:
        forecasts["AutoARIMA"] = np.repeat(np.nan, test_len)

    return forecasts

# ==========================================================
# Train / test split by store
# ==========================================================
# We use all rows before the 42-day holdout period as training.
# The forecast origin is the day before the first test date, so no extra day is removed.

forecast_horizon = 42
results = []

for store_id, store_df in sales.groupby("store_id"):
    store_df = store_df.sort_values("date").copy()
    store_df["sales"] = pd.to_numeric(store_df["sales"], errors="coerce")
    store_df = store_df.dropna(subset=["sales"])

    if len(store_df) < forecast_horizon + 5:
        continue

    train_series = store_df.iloc[:-forecast_horizon]["sales"].astype(float)
    test_series = store_df.iloc[-forecast_horizon:]["sales"].astype(float)
    test_dates = store_df.iloc[-forecast_horizon:]["date"]

    forecasts = build_forecasts(train_series, forecast_horizon)

    for model_name, pred in forecasts.items():
        pred = np.asarray(pred, dtype=float)
        if np.isnan(pred).any():
            continue

        results.append({
            "store_id": store_id,
            "model": model_name,
            "mae": mae(test_series.values, pred),
            "mape": mape(test_series.values, pred),
            "forecast_dates": json.dumps([d.strftime("%Y-%m-%d") for d in test_dates]),
            "actual_values": json.dumps([float(v) for v in test_series.values]),
            "forecast_values": json.dumps([float(v) for v in pred]),
        })

results_df = pd.DataFrame(results)
print(results_df.head())
print(f"\nTotal rows: {len(results_df)}")

# ==========================================================
# Save results
# ==========================================================

output_path = processed_dir / "benchmark_results_per_store.csv"

results_df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")

# ==========================================================
# Summary by model
# ==========================================================
summary_df = (
    results_df.groupby("model")[["mae", "mape"]]
    .mean()
    .sort_values("mae")
)
print(summary_df)

# ==========================================================
# Mean & Median Accuracy by Model
# ==========================================================
from pathlib import Path
import pandas as pd

results_df = pd.read_csv(
    Path("data/processed") / "benchmark_results_per_store.csv"
)

summary_stats = (
    results_df
    .groupby("model")
    .agg(
        MAE_Mean=("mae", "mean"),
        MAE_Median=("mae", "median"),
        MAPE_Mean=("mape", "mean"),
        MAPE_Median=("mape", "median"),
    )
    .round(3)
    .sort_values("MAPE_Mean")
)

print("\n============================================================")
print("Mean and Median Accuracy by Model")
print("============================================================")
print(summary_stats)
# ==========================================================
# ETS Benchmark Forecasting
# ==========================================================
#Since autoarima takes a lot of time, we also make ETS to balance efficiency and accuracy
# This cell adds ETS results without rerunning AutoARIMA.
# It uses the same train/test split, metrics, and result format as the benchmark cell.

#Accordind to EDA STEP18, Exploratory analysis revealed a clear weekly seasonal pattern across the dataset.
#Therefore, a seasonal period of seven days was adopted.
#Although no strong global trend was observed after aggregation, individual stores may still exhibit local trends.
#Hence, an additive trend component was retained and its suitability was validated through benchmark comparison.

from statsmodels.tsa.holtwinters import ExponentialSmoothing

ets_results = []

for store_id, store_df in sales.groupby("store_id"):
    store_df = store_df.sort_values("date").copy()
    store_df["sales"] = pd.to_numeric(store_df["sales"], errors="coerce")
    store_df = store_df.dropna(subset=["sales"])

    if len(store_df) < forecast_horizon + 14:
        continue

    train_series = store_df.iloc[:-forecast_horizon]["sales"].astype(float)
    test_series = store_df.iloc[-forecast_horizon:]["sales"].astype(float)
    test_dates = store_df.iloc[-forecast_horizon:]["date"]

    try:
        ets_model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated"
        ).fit(optimized=True)

        pred = ets_model.forecast(forecast_horizon)
        pred = np.maximum(np.asarray(pred, dtype=float), 0)

    except Exception:
        continue

    ets_results.append({
        "store_id": store_id,
        "model": "ETS",
        "mae": mae(test_series.values, pred),
        "mape": mape(test_series.values, pred),
        "forecast_dates": json.dumps([d.strftime("%Y-%m-%d") for d in test_dates]),
        "actual_values": json.dumps([float(v) for v in test_series.values]),
        "forecast_values": json.dumps([float(v) for v in pred]),
    })

ets_results_df = pd.DataFrame(ets_results)

# Remove old ETS rows if this cell is rerun, then append the new ETS results
results_df = results_df[results_df["model"] != "ETS"].copy()
results_df = pd.concat([results_df, ets_results_df], ignore_index=True)

print(ets_results_df.head())
print(f"\nETS rows added: {len(ets_results_df)}")

# ==========================================================
# Save updated benchmark results
# ==========================================================
output_path = output_dir / "benchmark_results_per_store_with_ets.csv"
results_df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")

# ETS results for all stores
ets_results_df = pd.DataFrame(ets_results)

print(ets_results_df)
print(f"\nNumber of stores forecasted: {len(ets_results_df)}")

# Save ETS results only
ets_output = output_dir / "ets_results_per_store.csv"
ets_results_df.to_csv(ets_output, index=False)

print(f"Saved: {ets_output}")
from pathlib import Path
import pandas as pd

# ============================================================
# File paths
# ============================================================

processed_dir = Path("data/processed")
raw_dir = Path("data/raw")

# ============================================================
# Load data
# ============================================================

results = pd.read_csv(processed_dir / "benchmark_results_per_store_with_ets.csv")
metadata = pd.read_csv(raw_dir / "metadata.csv")
# ============================================================
# Merge store type
# ============================================================

results = results.merge(
    metadata[["store_id", "store_type"]],
    on="store_id",
    how="left"
)

# ============================================================
# Average performance by Store Type and Model
# ============================================================

storetype_summary = (
    results
    .groupby(["store_type", "model"], as_index=False)
    .agg(
        Average_MAPE=("mape", "mean"),
        Average_MAE=("mae", "mean"),
        Number_of_Stores=("store_id", "nunique")
    )
)

# ============================================================
# Best model for each Store Type
# ============================================================

best_by_storetype = (
    storetype_summary
    .sort_values(["store_type", "Average_MAPE"])
    .groupby("store_type", as_index=False)
    .first()
)

# ============================================================
# Overall average performance
# ============================================================

overall_summary = (
    results
    .groupby("model", as_index=False)
    .agg(
        Average_MAPE=("mape", "mean"),
        Average_MAE=("mae", "mean"),
        Number_of_Stores=("store_id", "nunique")
    )
    .sort_values("Average_MAPE")
)

overall_best = overall_summary.head(1)

# ============================================================
# Print results
# ============================================================

print("=" * 60)
print("Best model for each Store Type")
print("=" * 60)
print(best_by_storetype)

print()

print("=" * 60)
print("Overall Best Model")
print("=" * 60)
print(overall_best)

# ============================================================
# Save results
# ============================================================

storetype_summary.to_csv(
    "average_performance_by_storetype.csv",
    index=False
)


overall_summary.to_csv(
    "overall_model_performance.csv",
    index=False
)



print("\nSaved:")
print("average_performance_by_storetype.csv")
print("overall_model_performance.csv")

# ============================================================
# ETS Mean & Median Accuracy
# ============================================================

ets_summary = (
    results[results["model"] == "ETS"]
    .groupby("model")[["mae", "mape"]]
    .agg(["mean", "median"])
    .round(3)
)

ets_summary.columns = [
    "MAE_Mean",
    "MAE_Median",
    "MAPE_Mean",
    "MAPE_Median",
]

print()
print("=" * 60)
print("ETS Accuracy Summary")
print("=" * 60)
print(ets_summary)

# ============================================================
# Validation: Winner Count by Store
# ============================================================

# Find the best model (lowest MAPE) for each store
best_model_each_store = (
    results
    .sort_values(["store_id", "mape"])
    .groupby("store_id", as_index=False)
    .first()
)

# Count how many stores each model wins
winner_summary = (
    best_model_each_store
    .groupby("model", as_index=False)
    .agg(
        Stores_Won=("store_id", "count")
    )
    .sort_values("Stores_Won", ascending=False)
)

# Percentage of stores won
winner_summary["Win_Rate (%)"] = (
    winner_summary["Stores_Won"] /
    winner_summary["Stores_Won"].sum() * 100
).round(2)

print()
print("=" * 60)
print("Validation: Best Model by Individual Store")
print("=" * 60)
print(winner_summary)


# ==========================================================
# ETS Final Sales Forecast
# ==========================================================
# Train one ETS model for each store using the full historical
# dataset and forecast daily sales for the next 6 weeks.

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================================
# File paths
# ==========================================================

processed_dir = Path("data/processed")

if not processed_dir.exists():
    processed_dir = Path.cwd().parent / "data" / "processed"

sales_path = processed_dir / "sales_clean.csv"
output_path = processed_dir / "ETS Result.csv"

# ==========================================================
# Load data
# ==========================================================

sales = pd.read_csv(sales_path)

sales["date"] = pd.to_datetime(sales["date"])
sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")

sales = (
    sales
    .dropna(subset=["store_id", "date", "sales"])
    .sort_values(["store_id", "date"])
)

# ==========================================================
# ETS forecast for all stores
# ==========================================================

forecast_horizon = 42
ets_results = []

for store_id, store_df in sales.groupby("store_id"):

    store_df = store_df.sort_values("date")
    train_series = store_df["sales"].astype(float)

    if len(train_series) < 14:
        continue

    try:
        ets_model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated"
        ).fit(optimized=True)

        forecast_values = ets_model.forecast(forecast_horizon)
        forecast_values = np.maximum(
            np.asarray(forecast_values, dtype=float),
            0
        )

        forecast_dates = pd.date_range(
            start=store_df["date"].max() + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq="D"
        )

        for forecast_date, forecast_sales in zip(
            forecast_dates,
            forecast_values
        ):
            ets_results.append({
                "store_id": store_id,
                "date": forecast_date,
                "sales": float(forecast_sales)
            })

    except Exception as error:
        print(f"ETS failed for {store_id}: {error}")

# ==========================================================
# Save forecast results
# ==========================================================

ets_result_df = pd.DataFrame(ets_results)

if ets_result_df.empty:
    raise RuntimeError("ETS did not produce forecasts for any store.")

ets_result_df = ets_result_df.sort_values(
    ["store_id", "date"]
).reset_index(drop=True)

ets_result_df.to_csv(
    output_path,
    index=False,
    date_format="%Y-%m-%d"
)

# ==========================================================
# Total sales result
# ==========================================================

number_of_stores = ets_result_df["store_id"].nunique()
total_sales_result = ets_result_df["sales"].sum()

print("=" * 60)
print("ETS Final Forecast")
print("=" * 60)
print(f"Stores forecasted     : {number_of_stores}")
print(f"Forecast days         : {forecast_horizon}")
print(f"Total forecast rows   : {len(ets_result_df)}")
print(f"Total sales result    : {total_sales_result:,.2f}")
print(f"Saved                 : {output_path}")



# ==========================================================
# Baseline Global Random Forest (Without Customer Features)
# ==========================================================

# This model predicts future sales using historical sales,
# time-series features, promotional variables, and store
# characteristics only. No customer-related features are
# included. The resulting forecasts serve as the baseline
# for evaluating the contribution of customer forecasting
# in the proposed two-stage framework.

sales = pd.read_csv(
    processed_dir / "sales_clean.csv",
    parse_dates=["date"],
    low_memory=False
)

future = pd.read_csv(
    processed_dir / "future_clean.csv",
    parse_dates=["date"],
    low_memory=False
)
# ==========================================================
# Remove unavailable feature
# ==========================================================

sales = sales.drop(columns=["customers"])
future = future.drop(columns=["customers"])
# ==========================================================
# Sort
# ==========================================================

sales = sales.sort_values(
    ["store_id", "date"]
).reset_index(drop=True)

future = future.sort_values(
    ["store_id", "date"]
).reset_index(drop=True)
# ==========================================================
# Rename for MLForecast
# ==========================================================

sales = sales.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds",
        "sales": "y"
    }
)

future = future.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds"
    }
)
# ==========================================================
# Rename Columns for MLForecast
# ==========================================================

sales = sales.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds",
        "sales": "y"
    }
)

future = future.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds"
    }
)

# ==========================================================
# Encode Categorical Features
# ==========================================================

state_holiday_map = {
    "0": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    0: 0,
    0.0: 0
}

store_type_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3
}

assortment_map = {
    "a": 0,
    "b": 1,
    "c": 2
}

for df in [sales, future]:
    df["state_holiday"] = (
        df["state_holiday"]
        .replace(state_holiday_map)
        .astype(int)
    )

    df["store_type"] = (
        df["store_type"]
        .map(store_type_map)
        .astype(int)
    )

    df["assortment"] = (
        df["assortment"]
        .map(assortment_map)
        .astype(int)
    )
# ==========================================================
# Create Additional Dynamic Features
# ==========================================================
# These features are known in advance and can be used as
# dynamic covariates during forecasting.
# ==========================================================

# Weekend indicator
sales["is_weekend"] = (
        sales["ds"].dt.dayofweek >= 5
).astype(int)

future["is_weekend"] = (
        future["ds"].dt.dayofweek >= 5
).astype(int)

# Quarter of the year
sales["quarter"] = sales["ds"].dt.quarter

future["quarter"] = future["ds"].dt.quarter

# ISO week number
# Represents the week of the year to capture
# recurring seasonal patterns throughout the year.
sales["week_of_year"] = (
    sales["ds"]
    .dt.isocalendar()
    .week
    .astype(int)
)

future["week_of_year"] = (
    future["ds"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Promotion during weekends
sales["promo_weekend"] = (
        sales["promo"] *
        sales["is_weekend"]
)

future["promo_weekend"] = (
        future["promo"] *
        future["is_weekend"]
)

print("=" * 60)
print("Additional Features")
print("=" * 60)

print(sales[
          [
              "unique_id",
              "ds",
              "promo",
              "is_weekend",
              "quarter",
              "week_of_year",
              "promo_weekend"
          ]
      ].head())
# ==========================================================
# Hyperparameter Grid
# ==========================================================
# A small hyperparameter grid is evaluated to balance forecast
# accuracy and computational efficiency.
#
# - n_estimators:
#   200 and 300 trees are tested to compare predictive performance
#   while keeping the training time manageable.
#
# - max_features:
#   "sqrt" is selected because it is the standard choice for Random
#   Forest regression and helps reduce correlation between trees.
#
# - min_samples_leaf:
#   Values of 1 and 5 are compared to evaluate the trade-off between
#   model complexity and generalisation. A larger leaf size may reduce
#   overfitting and improve robustness.
#
# - max_depth:
#   Unlimited tree depth (None) is used, while model complexity is
#   primarily controlled through the minimum leaf size.

# Rather than conducting an exhaustive grid search, a limited set of
# representative Random Forest hyperparameter combinations was evaluated.
# This approach balances forecast accuracy, computational efficiency,
# and the runtime required for Rolling Forecast Origin Cross-Validation.
parameter_grid = [

    {
        "name": "RF200_SQRT",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "name": "RF300_SQRT_Leaf5",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    }

]

# ==========================================================
# Hyperparameter Tuning using Rolling Forecast Origin CV
# ==========================================================

results = []

for params in parameter_grid:
    print("=" * 60)
    print(params["name"])
    print("=" * 60)

    model = RandomForestRegressor(

        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1

    )
# ------------------------------------------------------
# Configure MLForecast
# ------------------------------------------------------
# MLForecast is used to automatically generate time-series
# features for the global Random Forest model.
#
# - Lag features (1, 7, 14, 21 and 28 days) capture both
#   recent sales information and the strong weekly seasonal
#   pattern identified during the exploratory data analysis.
#
# - Rolling mean features summarise recent sales behaviour
#   and reduce the influence of short-term fluctuations.
#
# - Calendar features (day of week and month) allow the model
#   to learn recurring weekly and monthly seasonal effects.
    fcst = MLForecast(

        models=[model],

        freq="D",

        lags=[
            1,
            7,
            14,
            21,
            28
        ],

        lag_transforms={

            7: [
                RollingMean(window_size=4)
            ],

            28: [
                RollingMean(window_size=2)
            ]

        },

        date_features=[
            "dayofweek",
            "month"
        ]

    )
# ------------------------------------------------------
# Rolling Forecast Origin Cross-Validation
# ------------------------------------------------------
# Evaluate each hyperparameter combination using rolling
# forecast origin cross-validation.
#
# A 42-day forecast horizon is used to match the final
# forecasting task, while three rolling windows provide
# multiple validation periods for a more reliable estimate
# of model performance. The model is not refitted between
# windows to reduce computational cost during model selection.

    cv = fcst.cross_validation(

        df=sales,

        h=42,

        n_windows=3,

        refit=False,

        static_features=[]

    )

    prediction_col = cv.columns[-1]

    valid = cv["y"] > 0

    mae = mean_absolute_error(
        cv.loc[valid, "y"],
        cv.loc[valid, prediction_col]
    )

    rmse = np.sqrt(
        mean_squared_error(
            cv.loc[valid, "y"],
            cv.loc[valid, prediction_col]
        )
    )

    mape = (
               np.abs(
                   (
                           cv.loc[valid, "y"]
                           - cv.loc[valid, prediction_col]
                   )
                   / cv.loc[valid, "y"]
               )
           ).mean() * 100

    results.append({

        "Model": params["name"],
        "Trees": params["n_estimators"],
        "Leaf": params["min_samples_leaf"],
        "Max Features": params["max_features"],
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape

    })

    print(f"MAE :  {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"MAPE:  {mape:.2f}%")

# ==========================================================
# Hyperparameter Tuning Results
# ==========================================================

results = pd.DataFrame(results)

results = results.sort_values(
    by="MAPE",
    ascending=True
)

print(results)

results.to_csv(
    processed_dir / "rf_cv_results.csv",
    index=False
)

# ==========================================================
# Select Best Hyperparameters
# ==========================================================

best_result = results.iloc[0]

best_model = RandomForestRegressor(

    n_estimators=int(best_result["Trees"]),

    min_samples_leaf=int(best_result["Leaf"]),

    max_features=best_result["Max Features"],

    random_state=42,

    n_jobs=-1

)

# ==========================================================
# Train Final Model
# ==========================================================

final_fcst = MLForecast(

    models=[best_model],

    freq="D",

    lags=[
        1,
        7,
        14,
        21,
        28
    ],

    lag_transforms={

        7: [
            RollingMean(window_size=4)
        ],

        28: [
            RollingMean(window_size=2)
        ]

    },

    date_features=[
        "dayofweek",
        "month"
    ]

)

final_fcst.fit(

    sales,

    static_features=[]

)

print("=" * 60)
print("Best Model")
print("=" * 60)
print(best_result)
print()
print("Final model trained successfully.")
# ==========================================================
# Generate Features for the Final Model
# ==========================================================

features = final_fcst.preprocess(

    sales,

    static_features=[]

)

print("=" * 60)
print("Generated Features")
print("=" * 60)

print(features.head())

print(features.shape)

# ==========================================================
# Create Evaluation Dataset
# ==========================================================

eval_size = 42 * sales["unique_id"].nunique()

train_features = features.iloc[:-eval_size].copy()

test_features = features.iloc[-eval_size:].copy()

# Randomly sample observations to speed up permutation importance
test_features = test_features.sample(

    n=min(5000, len(test_features)),

    random_state=42

)

print("=" * 60)
print("Train Features:", train_features.shape)
print("Test Features :", test_features.shape)

# ==========================================================
# Customer Forecasting
# ==========================================================
# This section builds the first-stage Global Random Forest model
# to predict customer counts for the next 42 days.
#
# Historical customer observations, together with time-series,
# calendar, promotional, and store-related features, are used
# to generate future customer forecasts through recursive prediction.
#
# The resulting customer forecasts are saved and later used as
# input features for the second-stage sales forecasting model.

import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Load Data
# ==========================================================

sales = pd.read_csv(
    processed_dir / "sales_clean.csv",
    parse_dates=["date"],
    low_memory=False
)

future = pd.read_csv(
    processed_dir / "future_clean.csv",
    parse_dates=["date"],
    low_memory=False
)
# ==========================================================
# Remove Unavailable Feature
# ==========================================================
# Sales are unavailable when forecasting customers.

sales = sales.drop(columns=["sales"])

# ==========================================================
# Sort Data
# ==========================================================

sales = sales.sort_values(
    ["store_id", "date"]
).reset_index(drop=True)

future = future.sort_values(
    ["store_id", "date"]
).reset_index(drop=True)

# ==========================================================
# Rename Columns for MLForecast
# ==========================================================

sales = sales.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds",
        "customers": "y"
    }
)

future = future.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds"
    }
)

# ==========================================================
# Remove Target from Future Data
# ==========================================================

future = future.drop(columns=["customers"])

# ==========================================================
# Encode Categorical Features
# ==========================================================

state_holiday_map = {
    "0": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    0: 0,
    0.0: 0
}

store_type_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3
}

assortment_map = {
    "a": 0,
    "b": 1,
    "c": 2
}

for df in [sales, future]:
    df["state_holiday"] = (
        df["state_holiday"]
        .replace(state_holiday_map)
        .astype(int)
    )

    df["store_type"] = (
        df["store_type"]
        .map(store_type_map)
        .astype(int)
    )

    df["assortment"] = (
        df["assortment"]
        .map(assortment_map)
        .astype(int)
    )

# ==========================================================
# Create Additional Dynamic Features
# ==========================================================

# Weekend indicator
sales["is_weekend"] = (
        sales["ds"].dt.dayofweek >= 5
).astype(int)

future["is_weekend"] = (
        future["ds"].dt.dayofweek >= 5
).astype(int)

# Quarter
sales["quarter"] = sales["ds"].dt.quarter
future["quarter"] = future["ds"].dt.quarter

# ISO week number
sales["week_of_year"] = (
    sales["ds"]
    .dt.isocalendar()
    .week
    .astype(int)
)

future["week_of_year"] = (
    future["ds"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Promotion during weekends
sales["promo_weekend"] = (
        sales["promo"] *
        sales["is_weekend"]
)

future["promo_weekend"] = (
        future["promo"] *
        future["is_weekend"]
)


# ==========================================================
# Hyperparameter Grid
# ==========================================================
# The same hyperparameter grid as the sales forecasting(with and without customer prediction)
# model is used for consistency across experiments.

parameter_grid = [

    {
        "name": "RF200_SQRT",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "name": "RF300_SQRT_Leaf5",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    }

]

# ==========================================================
# Hyperparameter Tuning with Rolling Cross Validation
# ==========================================================

results = []

for params in parameter_grid:
    print("=" * 60)
    print(params["name"])
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1,

    )
# ------------------------------------------------------
# Rolling Forecast Origin Cross-Validation
# ------------------------------------------------------
# The same rolling forecast origin cross-validation
# configuration (42-day forecast horizon and three
# rolling validation windows) is adopted to ensure
# consistency and a fair comparison with the sales
# forecasting model.
    fcst = MLForecast(

        models=[model],

        freq="D",

        lags=[
            1,
            7,
            14
        ],

        lag_transforms={
            7: [
                RollingMean(window_size=4)
            ]
        },

        date_features=[
            "dayofweek",
            "month"
        ]

    )

    cv = fcst.cross_validation(

        df=sales,

        h=42,

        n_windows=3,

        refit=False,

        static_features=[]

    )

    prediction_col = cv.columns[-1]

    valid = cv["y"] > 0

    mae = mean_absolute_error(
        cv.loc[valid, "y"],
        cv.loc[valid, prediction_col]
    )

    rmse = np.sqrt(
        mean_squared_error(
            cv.loc[valid, "y"],
            cv.loc[valid, prediction_col]
        )
    )

    mape = (
            np.abs(
                (
                        cv.loc[valid, "y"]
                        - cv.loc[valid, prediction_col]
                )
                /
                cv.loc[valid, "y"]
            ).mean()
            * 100
    )

    results.append({

        "Model": params["name"],

        "Trees": params["n_estimators"],

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape

    })

    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

# ==========================================================
# Save Cross Validation Results
# ==========================================================

results = pd.DataFrame(results)

results = results.sort_values(
    by="MAPE",
    ascending=True
)

print(results)

results.to_csv(
    processed_dir / "customer_rf_cv_results.csv",
    index=False
)
# ==========================================================
# Select Best Hyperparameters
# ==========================================================

best_result = results.loc[
    results["MAPE"].idxmin()
]

best_trees = int(best_result["Trees"])

print("=" * 60)
print("Best Hyperparameters")
print("=" * 60)

print(best_result)

# ==========================================================
# Keep First 42 Forecast Days
# ==========================================================

future_42 = (
    future
    .groupby("unique_id", group_keys=False)
    .head(42)
)

print(future_42.shape)

# ==========================================================
# Build Final Forecast Model
# ==========================================================

final_model = RandomForestRegressor(

    n_estimators=best_trees,

    min_samples_leaf=1,

    random_state=42,

    n_jobs=-1

)

final_fcst = MLForecast(

    models=[final_model],

    freq="D",

    lags=[
        1,
        7,
        14
    ],

    lag_transforms={
        7: [
            RollingMean(window_size=4)
        ]
    },

    date_features=[
        "dayofweek",
        "month"
    ]

)

# ==========================================================
# Train Final Model on Full Historical Data
# ==========================================================

final_fcst.fit(

    sales,

    static_features=[]

)

print("Final model trained.")

# ==========================================================
# Forecast Future Customers
# ==========================================================

customer_forecast = final_fcst.predict(

    h=42,

    X_df=future_42

)

prediction_col = customer_forecast.columns[-1]

customer_forecast = customer_forecast.rename(

    columns={
        prediction_col: "customer_prediction"
    }

)

print("=" * 60)
print("Customer Forecast")
print("=" * 60)

print(customer_forecast.head())

print(customer_forecast.shape)

# ==========================================================
# Merge Forecast with Future Data
# ==========================================================

future_customer = future_42.merge(

    customer_forecast,

    on=[
        "unique_id",
        "ds"
    ],

    how="left"

)

print("=" * 60)
print("Missing Predictions")
print("=" * 60)

print(
    future_customer["customer_prediction"]
    .isna()
    .sum()
)

print()

print(future_customer.head())

# ==========================================================
# Save Customer Forecast
# ==========================================================

customer_forecast = future_customer[
    [
        "unique_id",
        "ds",
        "customer_prediction"
    ]
].copy()

output_file = processed_dir / "future_customer_prediction_rf.csv"

customer_forecast.to_csv(

    output_file,

    index=False

)

# ==========================================================
# Sales Forecasting
# ==========================================================
# This section builds the second-stage Global Random Forest model
# to forecast daily sales for all stores.
#
# The model combines historical sales information, temporal
# features, promotional variables, store characteristics,
# and the customer forecasts generated in the first stage.
# Customer-derived features are also included to capture
# the relationship between customer demand and future sales.
#
# The final model generates 42-day sales forecasts for all
# stores and saves the results for subsequent business analysis.

import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Load Data
# ==========================================================

sales = pd.read_csv(
    processed_dir / "sales_clean.csv",
    parse_dates=["date"],
    low_memory=False
)

future = pd.read_csv(
    processed_dir / "future_clean.csv",
    parse_dates=["date"],
    low_memory=False
)

# ==========================================================
# Load Customer Forecast
# ==========================================================

customer_forecast = pd.read_csv(
    processed_dir / "future_customer_prediction_rf.csv",
    parse_dates=["ds"],
    low_memory=False
)

metadata = pd.read_csv(
    raw_dir / "metadata.csv"
)
# ==========================================================
# Rename Columns for MLForecast
# ==========================================================

sales = sales.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds",
        "sales": "y",
        "customers": "customer_prediction"
    }
)

future = future.rename(
    columns={
        "store_id": "unique_id",
        "date": "ds"
    }
)

# ==========================================================
# Merge Future Customer Forecast
# ==========================================================

future = future.merge(

    customer_forecast,

    on=[
        "unique_id",
        "ds"
    ],

    how="left"

)

# ==========================================================
# Check Customer Prediction
# ==========================================================

print("=" * 60)
print("Customer Prediction")
print("=" * 60)

print(
    future[
        [
            "unique_id",
            "ds",
            "customer_prediction"
        ]
    ].head()
)

print()

print("Missing customer predictions:")

print(
    future["customer_prediction"]
    .isna()
    .sum()
)

# ==========================================================
# Encode Categorical Features
# ==========================================================

state_holiday_map = {
    "0": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    0: 0,
    0.0: 0
}

store_type_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3
}

assortment_map = {
    "a": 0,
    "b": 1,
    "c": 2
}

for df in [sales, future]:
    df["state_holiday"] = (
        df["state_holiday"]
        .replace(state_holiday_map)
        .fillna(0)
        .astype(int)
    )

    df["store_type"] = (
        df["store_type"]
        .map(store_type_map)
        .fillna(df["store_type"])
        .astype(int)
    )

    df["assortment"] = (
        df["assortment"]
        .map(assortment_map)
        .fillna(df["assortment"])
        .astype(int)
    )

# ==========================================================
# Create Additional Dynamic Features
# ==========================================================

# Weekend indicator
sales["is_weekend"] = (
        sales["ds"].dt.dayofweek >= 5
).astype(int)

future["is_weekend"] = (
        future["ds"].dt.dayofweek >= 5
).astype(int)

# Quarter
sales["quarter"] = sales["ds"].dt.quarter
future["quarter"] = future["ds"].dt.quarter

# ISO week number
sales["week_of_year"] = (
    sales["ds"]
    .dt.isocalendar()
    .week
    .astype(int)
)

future["week_of_year"] = (
    future["ds"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Promotion during weekends
sales["promo_weekend"] = (
        sales["promo"] *
        sales["is_weekend"]
)

future["promo_weekend"] = (
        future["promo"] *
        future["is_weekend"]
)

# Customer × Promotion
sales["customer_promo"] = (
        sales["customer_prediction"] *
        sales["promo"]
)

future["customer_promo"] = (
        future["customer_prediction"] *
        future["promo"]
)

# Customer × Weekend
sales["customer_weekend"] = (
        sales["customer_prediction"] *
        sales["is_weekend"]
)

future["customer_weekend"] = (
        future["customer_prediction"] *
        future["is_weekend"]
)

# Customer Ratio Feature
# Create a normalised customer feature by dividing the
# predicted customer count by the historical average
# customer count for each store.
#
# This feature represents the relative customer level
# rather than the absolute customer count, allowing the
# model to capture changes in customer demand while
# reducing differences in store size.
customer_mean = (
    sales
    .groupby("unique_id")["customer_prediction"]
    .mean()
)
sales["customer_ratio"] = (
        sales["customer_prediction"] /
        sales["unique_id"].map(customer_mean)
)

future["customer_ratio"] = (
        future["customer_prediction"] /
        future["unique_id"].map(customer_mean)
)

# ==========================================================
# Validate Features
# ==========================================================

print("=" * 60)
print("Additional Features")
print("=" * 60)

print(
    sales[
        [
            "unique_id",
            "ds",
            "customer_prediction",
            "promo",
            "customer_promo",
            "customer_weekend",
            "is_weekend",
            "quarter",
            "week_of_year",
            "promo_weekend"
        ]
    ].head()
)

print()

print(
    future[
        [
            "unique_id",
            "ds",
            "customer_prediction",
            "promo",
            "customer_promo",
            "customer_weekend",
            "is_weekend",
            "quarter",
            "week_of_year",
            "promo_weekend"
        ]
    ].head()
)

print()

print("=" * 60)
print("Missing Values")
print("=" * 60)

print(sales.isna().sum())

print()

print(future.isna().sum())
print(future.columns.tolist())
print(customer_forecast.head())
future2 = future.merge(
    customer_forecast,
    on=["unique_id", "ds"],
    how="left",
    indicator=True
)

print(future2["_merge"].value_counts())
print("=" * 60)
print("Features Used for Training")
print("=" * 60)
print(sales.columns.tolist())
# ==========================================================
# Hyperparameter Grid
# ==========================================================
# ==========================================================
# Hyperparameter Grid
# ==========================================================
# The same hyperparameter grid as the sales forecasting(without customer )and customer forecasting
# models are used for consistency across experiments.
parameter_grid = [

    {
        "name": "RF200_SQRT",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "name": "RF300_SQRT_Leaf5",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    }

]
# ------------------------------------------------------
# Rolling Forecast Origin Cross-Validation
# ------------------------------------------------------
# The same rolling forecast origin cross-validation
# configuration (42-day forecast horizon and three
# rolling validation windows) is adopted to ensure
# consistency and a fair comparison with the sales
# forecasting model.

results = []

for params in parameter_grid:
    print("=" * 60)
    print(params["name"])
    print("=" * 60)


    model = RandomForestRegressor(

        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1

    )


    fcst = MLForecast(

        models=[model],

        freq="D",

        lags=[
            1,
            7,
            14,
            21,
            28
        ],

        lag_transforms={

            7: [
                RollingMean(window_size=4)
            ],

            28: [
                RollingMean(window_size=2)
            ]

        },

        date_features=[
            "dayofweek",
            "month"
        ]

    )

    cv = fcst.cross_validation(

        df=sales,

        h=42,

        n_windows=3,

        refit=False,

        static_features=[]

    )

    cv.to_csv(

        processed_dir / f"{params['name']}_cv_predictions.csv",
        index=False
    )
    print("CV predictions saved.")

    prediction_col = cv.columns[-1]

    valid = cv["y"] > 0

    mae = mean_absolute_error(
        cv.loc[valid, "y"],
        cv.loc[valid, prediction_col]
    )

    rmse = np.sqrt(
        mean_squared_error(
            cv.loc[valid, "y"],
            cv.loc[valid, prediction_col]
        )
    )

    mape = (
               np.abs(
                   (
                           cv.loc[valid, "y"]
                           - cv.loc[valid, prediction_col]
                   )
                   / cv.loc[valid, "y"]
               )
           ).mean() * 100

    results.append({

        "Model": params["name"],
        "Trees": params["n_estimators"],
        "Leaf": params["min_samples_leaf"],
        "Max Features": params["max_features"],
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape

    })

    print(f"MAE :  {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"MAPE:  {mape:.2f}%")

# ==========================================================
# Hyperparameter Tuning Results
# ==========================================================

results = pd.DataFrame(results)

results = results.sort_values(
    by="MAPE",
    ascending=True
)

print(results)

results.to_csv(
    processed_dir / "rf_cv_results.csv",
    index=False
)

# ==========================================================
# Select Best Hyperparameters
# ==========================================================

best_result = results.iloc[0]
best_cv = pd.read_csv(

    processed_dir /
    f"{best_result['Model']}_cv_predictions.csv"

)

best_cv.to_csv(

    processed_dir / "rf_best_cv_predictions.csv",

    index=False

)

print("Best CV predictions saved.")

best_model = RandomForestRegressor(

    n_estimators=int(best_result["Trees"]),

    min_samples_leaf=int(best_result["Leaf"]),

    max_features=best_result["Max Features"],

    random_state=42,

    n_jobs=-1

)

# ==========================================================
# Train Final Model
# ==========================================================

final_fcst = MLForecast(

    models=[best_model],

    freq="D",

    lags=[
        1,
        7,
        14,
        21,
        28
    ],

    lag_transforms={

        7: [
            RollingMean(window_size=4)
        ],

        28: [
            RollingMean(window_size=2)
        ]

    },

    date_features=[
        "dayofweek",
        "month"
    ]

)

final_fcst.fit(

    sales,

    static_features=[]

)

print("=" * 60)
print("Best Model")
print("=" * 60)
print(best_result)
print()
print("Final model trained successfully.")
# ==========================================================
# Generate Features for the Final Model
# ==========================================================

features = final_fcst.preprocess(

    sales,

    static_features=[]

)

print("=" * 60)
print("Generated Features")
print("=" * 60)

print(features.head())

print(features.shape)

# ==========================================================
# Create Evaluation Dataset
# ==========================================================

eval_size = 42 * sales["unique_id"].nunique()

train_features = features.iloc[:-eval_size].copy()

test_features = features.iloc[-eval_size:].copy()

# Randomly sample observations to speed up permutation importance
test_features = test_features.sample(

    n=min(5000, len(test_features)),

    random_state=42

)

print("=" * 60)
print("Train Features:", train_features.shape)
print("Test Features :", test_features.shape)
# ==========================================================
# Permutation Feature Importance
# ==========================================================

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ==========================================================
# Generate Features from the Final Model
# ==========================================================

features = final_fcst.preprocess(

    sales,

    static_features=[]

)

print("=" * 60)
print("Generated Features")
print("=" * 60)

print(features.head())

print()

print("Feature Matrix Shape:")

print(features.shape)

# ==========================================================
# Create Evaluation Dataset
# ==========================================================

# Use the last 42 days for evaluation
eval_size = 42 * sales["unique_id"].nunique()

train_features = features.iloc[:-eval_size].copy()

test_features = features.iloc[-eval_size:].copy()

# Randomly sample observations to reduce computation time
test_features = test_features.sample(

    n=min(1000, len(test_features)),

    random_state=42

)

print("=" * 60)
print("Evaluation Dataset")
print("=" * 60)

print("Train Features:", train_features.shape)

print("Test Features :", test_features.shape)

# ==========================================================
# Prepare Test Data
# ==========================================================

X_test = test_features.drop(

    columns=[

        "unique_id",

        "ds",

        "y"

    ]

)

y_test = test_features["y"]

print()

print("X_test Shape:")

print(X_test.shape)

# ==========================================================
# Extract Final Random Forest
# ==========================================================

rf = list(final_fcst.models_.values())[0]

print("=" * 60)
print("Final Random Forest")
print("=" * 60)

print(rf)

# ==========================================================
# Permutation Importance
# ==========================================================

perm = permutation_importance(

    estimator=rf,

    X=X_test,

    y=y_test,

    scoring="neg_mean_absolute_error",

    n_repeats=2,

    random_state=42,

    n_jobs=1

)

print()

print("Permutation importance completed.")

# ==========================================================
# Create Importance Table
# ==========================================================

importance = pd.DataFrame({

    "Feature": X_test.columns,

    "Importance": perm.importances_mean

})

# ==========================================================
# Rename Feature for Visualization
# ==========================================================

importance["Feature"] = importance["Feature"].replace({

    "customer_prediction": "customer_count"

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

).reset_index(drop=True)

print("=" * 60)
print("Top 10 Important Features")
print("=" * 60)

print(importance.head(10))

# ==========================================================
# Plot Top 10 Features
# ==========================================================

top10 = importance.head(10)

plt.figure(

    figsize=(9, 6)

)

plt.barh(

    top10["Feature"],

    top10["Importance"]

)

plt.gca().invert_yaxis()

plt.xlabel(

    "Permutation Importance",

    fontsize=13

)

plt.ylabel(

    "Feature",

    fontsize=13

)

plt.title(

    "Top 10 Important Features",

    fontsize=15,

    fontweight="bold"

)

plt.xticks(

    fontsize=11

)

plt.yticks(

    fontsize=11

)

plt.tight_layout()

# ==========================================================
# Save Figure
# ==========================================================

plt.savefig(

    processed_dir / "permutation_feature_importance.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()

# ==========================================================
# Save Results
# ==========================================================

importance.to_csv(

    processed_dir / "permutation_importance.csv",

    index=False

)

print("=" * 60)
print("Permutation importance saved.")
print("=" * 60)
# ==========================================================
# Predict Future 42 Days
# ==========================================================

future_forecast = final_fcst.predict(

    h=42,

    X_df=future

)

prediction_col = future_forecast.columns[-1]

future_forecast = future_forecast.rename(

    columns={
        prediction_col: "sales_prediction"
    }

)

print("=" * 60)
print("Future Forecast")
print("=" * 60)

print(future_forecast.head())

print()

print("Forecast Shape:")

print(future_forecast.shape)

# ==========================================================
# Create Final Forecast Dataset
# ==========================================================

future_result = future[
    [
        "unique_id",
        "ds",
        "customer_prediction"
    ]
].merge(

    future_forecast,

    on=[
        "unique_id",
        "ds"
    ],

    how="left"

)

print()

print("=" * 60)
print("Final Forecast")
print("=" * 60)

print(future_result.head())

# ==========================================================
# Keep Forecast Until 2015-08-29
# ==========================================================

future_result = future_result[
    future_result["ds"] < "2015-08-30"
    ].copy()

print("=" * 60)
print("Forecast Period")
print("=" * 60)

print(
    future_result["ds"].min(),
    "to",
    future_result["ds"].max()
)

print(f"Rows: {len(future_result):,}")

# ==========================================================
# Save Forecast
# ==========================================================

output_file = processed_dir / "future_prediction_rf.csv"

future_result.to_csv(
    output_file,
    index=False
)

print("=" * 60)
print("Forecast saved successfully.")
print("=" * 60)

print(output_file)
# ==========================================================
# Weekly Forecast Aggregation
# ==========================================================
# Aggregate every consecutive 7 days into one forecast week
# ==========================================================

weekly_data = future_result.copy()

# ----------------------------------------------------------
# Create Forecast Week
# ----------------------------------------------------------

weekly_data = weekly_data.sort_values(
    ["unique_id", "ds"]
).reset_index(drop=True)

weekly_data["forecast_week"] = (

        weekly_data

        .groupby("unique_id")

        .cumcount()

        // 7

        + 1

)

print("=" * 60)
print("Forecast Weeks")
print("=" * 60)

print(
    weekly_data[
        [
            "unique_id",
            "ds",
            "forecast_week"
        ]
    ].head(15)
)

# ==========================================================
# Store Weekly Forecast
# ==========================================================

store_weekly = (

    weekly_data

    .groupby(

        [
            "unique_id",
            "forecast_week"
        ],

        as_index=False

    )["sales_prediction"]

    .sum()

)

print("=" * 60)
print("Store Weekly Forecast")
print("=" * 60)

print(store_weekly.head())

# ==========================================================
# Merge Metadata
# ==========================================================

metadata_rf = metadata.rename(

    columns={

        "store_id": "unique_id"

    }

)

store_weekly = store_weekly.merge(

    metadata_rf[
        [
            "unique_id",
            "store_type"
        ]
    ],

    how="left",

    on="unique_id"

)

# ==========================================================
# Store Type Weekly Forecast
# ==========================================================

store_type_weekly = (

    store_weekly

    .groupby(

        [
            "store_type",
            "forecast_week"
        ],

        as_index=False

    )["sales_prediction"]

    .sum()

)

print("=" * 60)
print("Store Type Weekly Forecast")
print("=" * 60)

print(store_type_weekly)

# ==========================================================
# Company Weekly Forecast
# ==========================================================

company_weekly = (

    store_weekly

    .groupby(

        "forecast_week",

        as_index=False

    )["sales_prediction"]

    .sum()

)

print("=" * 60)
print("Company Weekly Forecast")
print("=" * 60)

print(company_weekly)

# ==========================================================
# Total Forecast for the Next 6 Weeks
# ==========================================================

company_total = company_weekly["sales_prediction"].sum()

print("=" * 60)
print("Total Sales Forecast")
print("=" * 60)

print(f"6-Week Total Sales: {company_total:,.2f}")

# ==========================================================
# COO Focus:
# First Two Weeks
# ==========================================================

print("=" * 60)
print("Forecast for the Next Two Weeks")
print("=" * 60)

print(company_weekly.head(2))

# ==========================================================
# Save Results
# ==========================================================

store_weekly.to_csv(
    processed_dir / "store_weekly_forecast_rf.csv",
    index=False
)

store_type_weekly.to_csv(
    processed_dir / "store_type_weekly_forecast_rf.csv",
    index=False
)

company_weekly.to_csv(
    processed_dir / "company_weekly_forecast_rf.csv",
    index=False
)

print("=" * 60)
print("Weekly forecasts saved successfully.")
print("=" * 60)
#Where does the model perform poorly
# ==========================================================
# Forecast Accuracy by Store Type
# ==========================================================
# ==========================================================
# Forecast Accuracy by Store Type
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Load Cross Validation Predictions
# ==========================================================

cv = pd.read_csv(
    processed_dir / "rf_best_cv_predictions.csv",
    parse_dates=["ds"]
)

print("=" * 60)
print("Cross Validation Predictions")
print("=" * 60)

print(cv.head())

# ==========================================================
# Merge Store Type Information
# ==========================================================

store_info = (

    sales[
        [
            "unique_id",
            "store_type"
        ]
    ]

    .drop_duplicates()

)

cv = cv.merge(

    store_info,

    on="unique_id",

    how="left"

)

print("=" * 60)
print("Merged Store Type")
print("=" * 60)

print(cv.head())

# ==========================================================
# Prediction Column
# ==========================================================

prediction_col = cv.columns[-2]

print(f"Prediction column: {prediction_col}")

# ==========================================================
# Remove Zero Sales
# ==========================================================

cv = cv.loc[
    cv["y"] > 0
    ].copy()

# ==========================================================
# Absolute Percentage Error
# ==========================================================

cv["APE"] = (

                    np.abs(

                        cv["y"] -
                        cv[prediction_col]

                    )

                    /

                    cv["y"]

            ) * 100

# ==========================================================
# Forecast Accuracy by Store Type
# ==========================================================

accuracy = (

    cv

    .groupby(
        "store_type",
        as_index=False
    )

    .agg(

        MAPE=(
            "APE",
            "mean"
        ),

        Samples=(
            "APE",
            "count"
        )

    )

)

# ==========================================================
# Convert Store Labels
# ==========================================================

store_labels = {

    0: "A",
    1: "B",
    2: "C",
    3: "D"

}

accuracy["Store Type"] = (

    accuracy["store_type"]

    .map(store_labels)

)

accuracy = accuracy.sort_values(
    "Store Type"
)

print("=" * 60)
print("Forecast Accuracy")
print("=" * 60)

print(accuracy)

# ==========================================================
# Save Results
# ==========================================================

accuracy.to_csv(

    processed_dir /
    "forecast_accuracy_store_type.csv",

    index=False

)

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(

    accuracy["Store Type"],

    accuracy["MAPE"],

    color="#4F81BD",

    edgecolor="black",

    width=0.6

)

for bar in bars:
    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width() / 2,

        height + 0.1,

        f"{height:.2f}%",

        ha="center",

        fontsize=11,

        fontweight="bold"

    )

plt.title(

    "Forecast Accuracy by Store Type",

    fontsize=16,

    fontweight="bold"

)

plt.xlabel(

    "Store Type",

    fontsize=12

)

plt.ylabel(

    "MAPE (%)",

    fontsize=12

)

plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)

plt.tight_layout()

plt.savefig(

    processed_dir /
    "forecast_accuracy_store_type.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Forecast Accuracy by Promotion
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Load Cross Validation Predictions
# ----------------------------------------------------------

print("=" * 60)
print("Cross Validation Predictions")
print("=" * 60)

print(cv.head())

# ----------------------------------------------------------
# Merge Promotion Information
# ----------------------------------------------------------

promo_info = sales[
    [
        "unique_id",
        "ds",
        "promo"
    ]
].copy()

promo_info["ds"] = pd.to_datetime(promo_info["ds"])
cv["ds"] = pd.to_datetime(cv["ds"])

cv = cv.merge(

    promo_info,

    on=[
        "unique_id",
        "ds"
    ],

    how="left"

)

print("=" * 60)
print("Merged Promotion")
print("=" * 60)

print(cv.head())

# ----------------------------------------------------------
# Prediction Column
# ----------------------------------------------------------

prediction_col = "RandomForestRegressor"

print(f"Prediction column: {prediction_col}")

# ----------------------------------------------------------
# Remove Zero Sales
# ----------------------------------------------------------

cv = cv[

    cv["y"] > 0

    ].copy()

# ----------------------------------------------------------
# Calculate Absolute Percentage Error
# ----------------------------------------------------------

cv["APE"] = (

                    np.abs(

                        cv["y"] -

                        cv[prediction_col]

                    )

                    / cv["y"]

            ) * 100

# ----------------------------------------------------------
# Average MAPE
# ----------------------------------------------------------

promo_result = (

    cv

    .groupby("promo")

    .agg(

        MAPE=("APE", "mean"),

        Samples=("APE", "count")

    )

    .reset_index()

)

promo_result["Scenario"] = (

    promo_result["promo"]

    .map({

        0: "Non-Promo",

        1: "Promotion"

    })

)

print("=" * 60)
print("Forecast Accuracy")
print("=" * 60)

print(promo_result)

promo_result.to_csv(

    processed_dir / "forecast_accuracy_promo.csv",

    index=False

)

# ==========================================================
# Forecast Accuracy by Sales Volume
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Average Historical Sales of Each Store
# ----------------------------------------------------------

store_volume = (

    sales

    .groupby("unique_id")["y"]

    .mean()

    .reset_index()

    .rename(columns={"y": "avg_sales"})

)

# ----------------------------------------------------------
# Divide into Low / Medium / High
# ----------------------------------------------------------

store_volume["Sales Group"] = pd.qcut(

    store_volume["avg_sales"],

    q=3,

    labels=[

        "Low",

        "Medium",

        "High"

    ]

)

print("=" * 60)
print("Store Sales Groups")
print("=" * 60)

print(store_volume.head())

# ----------------------------------------------------------
# Create a Clean Copy of CV
# ----------------------------------------------------------

cv_volume = cv.copy()

# Remove columns from previous analyses if they exist
cv_volume = cv_volume.drop(

    columns=[
        "Sales Group",
        "Sales Group_x",
        "Sales Group_y"
    ],

    errors="ignore"

)

# ----------------------------------------------------------
# Merge Sales Groups
# ----------------------------------------------------------

cv_volume = cv_volume.merge(

    store_volume[
        [
            "unique_id",
            "Sales Group"
        ]
    ],

    on="unique_id",

    how="left"

)

print("=" * 60)
print("Merged Sales Group")
print("=" * 60)

print(cv_volume.head())

# ----------------------------------------------------------
# Prediction Column
# ----------------------------------------------------------

prediction_col = "RandomForestRegressor"

# ----------------------------------------------------------
# Remove Zero Sales
# ----------------------------------------------------------

cv_volume = cv_volume[

    cv_volume["y"] > 0

    ].copy()

# ----------------------------------------------------------
# Calculate Absolute Percentage Error
# ----------------------------------------------------------

cv_volume["APE"] = (

                           np.abs(

                               cv_volume["y"]

                               -

                               cv_volume[prediction_col]

                           )

                           /

                           cv_volume["y"]

                   ) * 100

# ----------------------------------------------------------
# Average MAPE by Sales Group
# ----------------------------------------------------------

volume_result = (

    cv_volume

    .groupby("Sales Group", observed=True)

    .agg(

        MAPE=("APE", "mean"),

        Samples=("APE", "count"),

        Average_Sales=("y", "mean")

    )

    .reset_index()

)

print("=" * 60)
print("Forecast Accuracy by Sales Volume")
print("=" * 60)

print(volume_result)

volume_result.to_csv(

    processed_dir /

    "forecast_accuracy_sales_volume.csv",

    index=False

)

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(8, 5.5))

bars = plt.bar(

    volume_result["Sales Group"],

    volume_result["MAPE"],

    color=[

        "#7FB3D5",

        "#5DADE2",

        "#2874A6"

    ],

    edgecolor="black",

    width=0.60

)

plt.title(

    "Forecast Accuracy by Sales Volume",

    fontsize=18,

    fontweight="bold",

    pad=18

)

plt.ylabel(

    "MAPE (%)",

    fontsize=13

)

plt.xlabel(

    "Store Sales Group",

    fontsize=13

)

plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)

# ----------------------------------------------------------
# Leave Space for Labels
# ----------------------------------------------------------

upper = volume_result["MAPE"].max() * 1.20

plt.ylim(

    0,

    upper

)

offset = volume_result["MAPE"].max() * 0.03

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

for bar, value, n in zip(

        bars,

        volume_result["MAPE"],

        volume_result["Samples"]

):
    plt.text(

        bar.get_x() + bar.get_width() / 2,

        value + offset,

        f"{value:.2f}%\n(n={n})",

        ha="center",

        va="bottom",

        fontsize=10,

        fontweight="bold"

    )

plt.tight_layout()

plt.savefig(

    processed_dir /

    "forecast_accuracy_sales_volume.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Forecast Accuracy by Forecast Horizon
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Copy Cross Validation Results
# ----------------------------------------------------------

cv_horizon = cv.copy()

prediction_col = "RandomForestRegressor"

# ----------------------------------------------------------
# Remove Zero Sales
# ----------------------------------------------------------

cv_horizon = cv_horizon[
    cv_horizon["y"] > 0
    ].copy()

# ----------------------------------------------------------
# Absolute Percentage Error
# ----------------------------------------------------------

cv_horizon["APE"] = (
                            np.abs(
                                cv_horizon["y"] -
                                cv_horizon[prediction_col]
                            )
                            /
                            cv_horizon["y"]
                    ) * 100

# ----------------------------------------------------------
# Forecast Horizon
# ----------------------------------------------------------

cv_horizon["cutoff"] = pd.to_datetime(cv_horizon["cutoff"])
cv_horizon["ds"] = pd.to_datetime(cv_horizon["ds"])

cv_horizon["Horizon"] = (
        cv_horizon["ds"] -
        cv_horizon["cutoff"]
).dt.days

print("=" * 60)
print("Forecast Horizons")
print("=" * 60)
print(cv_horizon["Horizon"].describe())

# ----------------------------------------------------------
# Weekly Horizon Groups
# ----------------------------------------------------------

bins = [0, 7, 14, 21, 28, 35, 42]

labels = [
    "Week 1",
    "Week 2",
    "Week 3",
    "Week 4",
    "Week 5",
    "Week 6"
]

cv_horizon["Forecast Week"] = pd.cut(

    cv_horizon["Horizon"],

    bins=bins,

    labels=labels,

    include_lowest=True

)

# ----------------------------------------------------------
# Average MAPE
# ----------------------------------------------------------

horizon_result = (

    cv_horizon

    .groupby(
        "Forecast Week",
        observed=True
    )

    .agg(

        MAPE=("APE", "mean"),

        Samples=("APE", "count")

    )

    .reset_index()

)

print("=" * 60)
print("Forecast Accuracy by Horizon")
print("=" * 60)

print(horizon_result)

horizon_result.to_csv(

    processed_dir /

    "forecast_accuracy_horizon.csv",

    index=False

)

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(9, 5.5))

plt.plot(

    horizon_result["Forecast Week"],

    horizon_result["MAPE"],

    marker="o",

    linewidth=3,

    markersize=9,

    color="#2F6DB2"

)

plt.grid(

    linestyle="--",

    alpha=0.3

)

plt.title(

    "Forecast Accuracy by Forecast Horizon",

    fontsize=18,

    fontweight="bold"

)
plt.axvspan(-0.3, 1.3,
            color="#FDE9D9",
            alpha=0.4,
            label="COO Focus")

plt.xlabel(

    "Forecast Week",

    fontsize=13

)

plt.ylabel(

    "MAPE (%)",

    fontsize=13

)

plt.ylim(

    horizon_result["MAPE"].min() - 0.3,

    horizon_result["MAPE"].max() + 0.6

)

offset = 0.08

for x, y, n in zip(

        horizon_result["Forecast Week"],

        horizon_result["MAPE"],

        horizon_result["Samples"]

):
    plt.text(

        x,

        y + offset,

        f"{y:.2f}%\n(n={n})",

        ha="center",

        fontsize=10,

        fontweight="bold"

    )

plt.tight_layout()

plt.savefig(

    processed_dir /

    "forecast_accuracy_horizon.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()

# ==========================================================
# Company Historical Sales
# ==========================================================

company_history = (

    sales
    .groupby("ds", as_index=False)["y"]
    .sum()

)

company_history["unique_id"] = "Company"

company_history = company_history[
    ["unique_id", "ds", "y"]
]

# Keep only the last 42 historical days
company_history = company_history.tail(42)

# ==========================================================
# Company Forecast
# ==========================================================

company_forecast = (

    future_result
    .groupby("ds", as_index=False)["sales_prediction"]
    .sum()

)

company_forecast["unique_id"] = "Company"

company_forecast = company_forecast.rename(
    columns={
        "sales_prediction": "RandomForest"
    }
)

company_forecast = company_forecast[
    ["unique_id", "ds", "RandomForest"]
]

# ==========================================================
# Plot
# ==========================================================
fig = StatsForecast.plot(

    df=company_history,

    forecasts_df=company_forecast

)

fig.savefig(

    processed_dir / "company_forecast.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Company Weekly Forecast
# ==========================================================

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Daily Company Forecast
# ----------------------------------------------------------

daily_company = (

    future_result

    .groupby(
        "ds",
        as_index=False
    )["sales_prediction"]

    .sum()

)

print("=" * 60)
print("Daily Company Forecast")
print("=" * 60)

print(daily_company.head())

print(daily_company.shape)

# ----------------------------------------------------------
# Weekly Forecast
# ----------------------------------------------------------

weekly_forecast = (

    daily_company

    .groupby(

        np.arange(len(daily_company)) // 7,

        as_index=False

    )["sales_prediction"]

    .sum()

)

weekly_forecast["week"] = np.arange(1, 7)

weekly_forecast["Sales_Million"] = (

        weekly_forecast["sales_prediction"] / 1_000_000

)

print("=" * 60)
print("Weekly Forecast")
print("=" * 60)

print(weekly_forecast)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

weekly_forecast.to_csv(

    processed_dir / "weekly_forecast_rf.csv",

    index=False

)

# ==========================================================
# Weekly Forecast Dashboard
# ==========================================================

import matplotlib.pyplot as plt

weekly_sales = weekly_forecast["Sales_Million"]

weeks = weekly_forecast["week"]

total_sales = weekly_sales.sum()

average_sales = weekly_sales.mean()

fig, ax = plt.subplots(
    figsize=(13, 6)
)

# ----------------------------------------------------------
# Background
# ----------------------------------------------------------

ax.axvspan(
    0.8,
    2.2,
    color="#FDE9D9",
    alpha=0.6,
    label="Weeks 1–2 (COO Focus)"
)

ax.axvspan(
    2.8,
    6.2,
    color="#EAF2FB",
    alpha=0.45,
    label="Weeks 3–6"
)

# ----------------------------------------------------------
# Forecast Line
# ----------------------------------------------------------

ax.plot(
    weeks,
    weekly_sales,
    color="#2F6DB2",
    linewidth=3,
    marker="o",
    markersize=10
)

# ----------------------------------------------------------
# Average Line
# ----------------------------------------------------------

ax.axhline(
    average_sales,
    color="gray",
    linestyle="--",
    linewidth=2,
    label="6-week Average"
)

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

offset = (
                 weekly_sales.max()
                 - weekly_sales.min()
         ) * 0.05

for x, y in zip(weeks, weekly_sales):
    ax.text(
        x,
        y + offset,
        f"{y:.2f}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

# ----------------------------------------------------------
# Statistics Box
# ----------------------------------------------------------

stats = (
    f"Total Forecast (6 Weeks): {total_sales:.2f} M\n"
    f"Average Weekly Sales: {average_sales:.2f} M"
)

ax.text(
    5.35,
    weekly_sales.max() + 0.25,
    stats,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(
        facecolor="white",
        edgecolor="gray",
        boxstyle="round,pad=0.4"
    )
)

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------

ax.set_title(
    "Projected Company Sales for the Next 6 Weeks",
    fontsize=20,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Forecast Week",
    fontsize=14
)

ax.set_ylabel(
    "Sales (Million)",
    fontsize=14
)

ax.set_xticks(weeks)

ax.grid(
    alpha=0.3,
    linestyle="--"
)

ax.legend(
    loc="upper left",
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    processed_dir / "kpi_total_sales_6weeks.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
# ==========================================================
# Top 5 Stores Forecast (Next 2 Weeks)
# ==========================================================

# ----------------------------------------------------------
# First 14 Forecast Days
# ----------------------------------------------------------

start_date = future_result["ds"].min()

end_date = start_date + pd.Timedelta(days=13)

future_14 = future_result[

    (future_result["ds"] >= start_date) &
    (future_result["ds"] <= end_date)

    ].copy()

# ----------------------------------------------------------
# Find Top 5 Stores (Next 2 Weeks)
# ----------------------------------------------------------

top5_stores = (

    future_14

    .groupby(
        "unique_id",
        as_index=False
    )["sales_prediction"]

    .sum()

    .sort_values(
        "sales_prediction",
        ascending=False
    )

    .head(5)

)

print("=" * 60)
print("Top 5 Stores (Next 2 Weeks)")
print("=" * 60)

print(top5_stores)

# ----------------------------------------------------------
# Historical Data
# ----------------------------------------------------------

history_top5 = (

    sales

    .merge(

        top5_stores[
            ["unique_id"]
        ],

        on="unique_id"

    )

)

history_top5 = history_top5[
    [
        "unique_id",
        "ds",
        "y"
    ]
]

# Keep only the last 42 historical days

history_top5 = (

    history_top5

    .sort_values(
        [
            "unique_id",
            "ds"
        ]
    )

    .groupby("unique_id")

    .tail(42)

)

# ----------------------------------------------------------
# Forecast Data (Next 2 Weeks Only)
# ----------------------------------------------------------

forecast_top5 = (

    future_14

    .merge(

        top5_stores[
            ["unique_id"]
        ],

        on="unique_id"

    )

)

forecast_top5 = forecast_top5.rename(

    columns={

        "sales_prediction": "RandomForest"

    }

)

forecast_top5 = forecast_top5[
    [
        "unique_id",
        "ds",
        "RandomForest"
    ]
]

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

import matplotlib.pyplot as plt

fig = StatsForecast.plot(

    df=history_top5,

    forecasts_df=forecast_top5

)

# ----------------------------------------------------------
# Improve Title
# ----------------------------------------------------------

fig.suptitle(

    "Top 5 Store Sales Forecast (Next 2 Weeks)",

    fontsize=18,

    fontweight="bold",

    y=1.02

)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

fig.savefig(

    processed_dir / "top5_store_forecast_2weeks.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Top 5 Stores Weekly Forecast
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Find Top 5 Stores
# ----------------------------------------------------------

top5_stores = (

    store_weekly

    .groupby(
        "unique_id",
        as_index=False
    )["sales_prediction"]

    .sum()

    .sort_values(
        "sales_prediction",
        ascending=False
    )

    .head(5)

)

print("=" * 60)
print("Top 5 Stores")
print("=" * 60)

print(top5_stores)

# ----------------------------------------------------------
# Weekly Forecast for Top 5 Stores
# ----------------------------------------------------------

top5_weekly = (

    store_weekly

    .merge(

        top5_stores[
            ["unique_id"]
        ],

        on="unique_id"

    )

)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(13, 6))

for store in top5_stores["unique_id"]:
    temp = (

        top5_weekly

        .loc[
            top5_weekly["unique_id"] == store
            ]

        .sort_values("forecast_week")

    )

    plt.plot(

        temp["forecast_week"],

        temp["sales_prediction"],

        marker="o",

        linewidth=2.5,

        markersize=8,

        label=store.replace("store_", "Store ")

    )

# ----------------------------------------------------------
# Highlight COO Focus
# ----------------------------------------------------------

plt.axvspan(

    0.8,

    2.2,

    color="#FDE9D9",

    alpha=0.5,

    label="Weeks 1–2 (COO Focus)"

)

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------

plt.title(

    "Top 5 Stores: Weekly Sales Forecast",

    fontsize=18,

    fontweight="bold"

)

plt.xlabel(

    "Forecast Week",

    fontsize=13

)

plt.ylabel(

    "Predicted Weekly Sales",

    fontsize=13

)

plt.xticks(range(1, 7))

plt.grid(

    linestyle="--",

    alpha=0.3

)

plt.legend(

    title="Store",

    fontsize=10,

    title_fontsize=11,

    loc="upper right"

)

plt.tight_layout()

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.savefig(

    processed_dir / "top5_store_weekly_forecast.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Store Type Weekly Forecast
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(13, 6))

store_types = sorted(
    store_type_weekly["store_type"].unique()
)

for store_type in store_types:
    temp = (

        store_type_weekly

        .loc[
            store_type_weekly["store_type"] == store_type
            ]

        .sort_values("forecast_week")

    )

    plt.plot(

        temp["forecast_week"],

        temp["sales_prediction"],

        marker="o",

        linewidth=2.5,

        markersize=8,

        label=f"Type {store_type.upper()}"

    )

# ----------------------------------------------------------
# COO Highlight
# ----------------------------------------------------------

plt.axvspan(

    0.8,

    2.2,

    color="#FDE9D9",

    alpha=0.45,

    label="Weeks 1–2 (COO Focus)"

)

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------

plt.title(

    "Weekly Sales Forecast by Store Type",

    fontsize=18,

    fontweight="bold"

)

plt.xlabel(

    "Forecast Week",

    fontsize=13

)

plt.ylabel(

    "Predicted Weekly Sales",

    fontsize=13

)

plt.xticks(range(1, 7))

plt.grid(

    linestyle="--",

    alpha=0.3

)

plt.legend(

    title="Store Type",

    fontsize=10,

    title_fontsize=11,

    loc="upper right"

)

plt.tight_layout()

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.savefig(

    processed_dir / "store_type_weekly_forecast.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Cumulative Sales Forecast
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Prepare Data
# ----------------------------------------------------------

cumulative_sales = company_weekly.copy()

cumulative_sales["cumulative_sales"] = (

    cumulative_sales["sales_prediction"]

    .cumsum()

)

cumulative_sales["Sales_Million"] = (

        cumulative_sales["cumulative_sales"]

        / 1_000_000

)

weeks = cumulative_sales["forecast_week"]

sales = cumulative_sales["Sales_Million"]

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, ax = plt.subplots(

    figsize=(13, 6)

)

ax.plot(

    weeks,

    sales,

    color="#2F6DB2",

    linewidth=3,

    marker="o",

    markersize=10

)

# ----------------------------------------------------------
# Value Labels
# ----------------------------------------------------------

offset = sales.max() * 0.02

for x, y in zip(weeks, sales):
    ax.text(

        x,

        y + offset,

        f"{y:.1f}",

        ha="center",

        fontsize=11,

        fontweight="bold"

    )

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------

ax.set_title(

    "Cumulative Sales Forecast",

    fontsize=20,

    fontweight="bold",

    pad=20

)

ax.set_xlabel(

    "Forecast Week",

    fontsize=14

)

ax.set_ylabel(

    "Cumulative Sales (Million)",

    fontsize=14

)

ax.set_xticks(range(1, 7))

ax.grid(

    linestyle="--",

    alpha=0.3

)

# ----------------------------------------------------------
# KPI Outside Figure
# ----------------------------------------------------------


plt.tight_layout(

    rect=[0, 0, 0.86, 1]

)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.savefig(

    processed_dir / "cumulative_sales_forecast.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# KPI Card
# Total Sales Forecast (Next 6 Weeks)
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# KPI
# ----------------------------------------------------------

total_sales = company_weekly["sales_prediction"].sum() / 1_000_000

# ----------------------------------------------------------
# Card
# ----------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 4.8)
)

ax.axis("off")

# Border
card = plt.Rectangle(
    (0.05, 0.08),
    0.90,
    0.84,
    fill=False,
    linewidth=2.8,
    edgecolor="#7A8FA6"  # 柔和灰蓝
)

ax.add_patch(card)

# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

ax.text(

    0.5,

    0.78,

    "TOTAL SALES FORECAST",

    ha="center",

    fontsize=24,

    fontweight="bold",

    color="#2E3A46"

)

ax.text(

    0.5,

    0.69,

    "NEXT 6 WEEKS",

    ha="center",

    fontsize=18,

    color="#6B7C93"

)

# ----------------------------------------------------------
# KPI Number
# ----------------------------------------------------------

ax.text(

    0.5,

    0.48,

    f"{total_sales:.1f} M",

    ha="center",

    fontsize=48,

    fontweight="bold",

    color="#4A90E2"

)

# ----------------------------------------------------------
# Subtitle
# ----------------------------------------------------------

ax.text(

    0.5,

    0.22,

    "Company-wide Forecast",

    ha="center",

    fontsize=18,

    color="#6B7C93"

)

plt.tight_layout()

plt.savefig(

    processed_dir / "weekly_forecast_dashboard.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()
# ==========================================================
# Top 20 Store Contribution Analysis
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# First 14 Forecast Days
# ----------------------------------------------------------

start_date = future_result["ds"].min()

end_date = start_date + pd.Timedelta(days=13)

future_14 = future_result[

    (future_result["ds"] >= start_date) &
    (future_result["ds"] <= end_date)

    ].copy()

print("=" * 60)
print("Forecast Period")
print("=" * 60)
print(start_date)
print(end_date)

# ----------------------------------------------------------
# Aggregate Forecast by Store
# ----------------------------------------------------------

pareto = (

    future_result

    .groupby(
        "unique_id",
        as_index=False
    )["sales_prediction"]

    .sum()

    .sort_values(
        "sales_prediction",
        ascending=False
    )

)

# ----------------------------------------------------------
# Company Total Forecast
# ----------------------------------------------------------

company_total = pareto["sales_prediction"].sum()

# ----------------------------------------------------------
# Top 20 Stores
# ----------------------------------------------------------

top_n = 20

pareto = pareto.head(top_n).copy()

pareto["label"] = (

    pareto["unique_id"]

    .str.replace("store_", "Store ")

)

# ----------------------------------------------------------
# Cumulative Contribution
# ----------------------------------------------------------

pareto["cum_sales"] = pareto["sales_prediction"].cumsum()

pareto["cum_percent"] = (

        pareto["cum_sales"]

        / company_total

        * 100

)

top20_share = (

        pareto["sales_prediction"].sum()

        / company_total

        * 100

)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, ax1 = plt.subplots(
    figsize=(14, 6)
)

# ----------------------------------------------------------
# Bar
# ----------------------------------------------------------

ax1.bar(

    pareto["label"],

    pareto["sales_prediction"] / 1_000_000,

    color="#4F81BD"

)

ax1.set_ylabel(

    "Forecast Sales (Million)",

    fontsize=13

)

ax1.set_xlabel(

    "Store",

    fontsize=13

)

ax1.tick_params(

    axis="x",

    rotation=60

)

# ----------------------------------------------------------
# Cumulative Contribution
# ----------------------------------------------------------

ax2 = ax1.twinx()

ax2.plot(

    pareto["label"],

    pareto["cum_percent"],

    color="#C0504D",

    marker="o",

    linewidth=2.5

)

ax2.set_ylabel(

    "Cumulative Contribution (%)",

    fontsize=13

)

ax2.set_ylim(0, 100)

# ----------------------------------------------------------
# 50% Reference Line
# ----------------------------------------------------------

ax2.axhline(

    50,

    color="gray",

    linestyle="--",

    linewidth=1.5

)

# ----------------------------------------------------------
# Summary Box
# ----------------------------------------------------------

ax2.text(

    0.82,

    0.80,

    f"Top {top_n} Stores\n"

    f"{top20_share:.1f}% of\n"

    f"Company Forecast",

    transform=ax2.transAxes,

    fontsize=12,

    bbox=dict(

        facecolor="white",

        edgecolor="gray",

        boxstyle="round,pad=0.4"

    )

)

# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

plt.title(

    f"Top {top_n} Store Contribution Analysis(Next 2 Weeks)",

    fontsize=18,

    fontweight="bold"

)

plt.tight_layout()

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.savefig(

    processed_dir / "top20_store_contribution.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()



#AI Declaration
#Used ChatGPT and Codex for code fixation and recommendation, better coding grammar way to approach the same goal and tidying up csv. production. We finally identified the result by ourselves.

#Used ChatGPT for fixation and checking the indentation, consulting whether our solution is practical in real world for some steps and checking English grammar.



