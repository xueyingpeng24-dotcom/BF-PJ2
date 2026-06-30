#Initial Data Inspection

# =============================================================================
# STEP 1: Import Required Libraries
# =============================================================================
# pandas is used for data manipulation and analysis.
# pathlib provides an operating-system independent way to construct file paths.

from pathlib import Path
import pandas as pd

# =============================================================================
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
# STEP 2: Read the Dataset
# =============================================================================
# The project assumes that all input files are stored in the same directory
# structure. Therefore, the file path is created dynamically using pathlib
# instead of hardcoding an absolute path.

raw_dir = Path("..") / "data" / "raw"
sales_data_path = raw_dir / "sales_data.csv"

# Read the sales dataset.
# low_memory=False prevents pandas from inferring column types in chunks,
# which avoids unnecessary DtypeWarning messages for larger datasets.

sales_data = pd.read_csv(
    sales_data_path,
    low_memory=False
)


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
# STEP 8: Verify Date Range
# =============================================================================
# Inspect the temporal coverage of the dataset.
# Knowing the start and end dates provides an overview of the available
# historical period and helps verify that the dataset has been loaded correctly.

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
# =============================================================================
# STEP 14: Store-Level Data Completeness
# =============================================================================
# Since the forecasting task is performed for individual stores,
# it is important to verify whether every store contains a similar amount of
# historical information.
#
# Large differences in the number of observations may indicate incomplete
# histories that require special treatment during model development.

# Count the number of unique dates available for each store.
# This verifies whether every store has a complete historical time series.

store_counts = (
    sales_data
    .groupby("store_id")["date"]
    .nunique()
)

print("\n" + "=" * 60)
print("STORE-LEVEL OBSERVATIONS")
print("=" * 60)
print(store_counts.describe())

print("\nNumber of stores:", store_counts.size)

print(
    "Stores with complete history:",
    (store_counts == store_counts.max()).sum()
)


# STEP 15: Store-Level Sales Summary
# =============================================================================
# Compare the average sales across stores to evaluate
# whether substantial heterogeneity exists between locations.

store_sales_summary = (
    sales_data
    .groupby("store_id")["sales"]
    .agg(mean_sales="mean")
)

print("\n" + "=" * 60)
print("STORE-LEVEL SALES SUMMARY")
print("=" * 60)

print(store_sales_summary["mean_sales"].describe())


print("\n" + "=" * 60)
print(f"Total stores : {store_counts.size}")
print(f"Total records: {len(sales_data)}")
print(f"Date range   : {sales_data['date'].min().date()} to {sales_data['date'].max().date()}")
print(f"Missing values: {sales_data.isnull().sum().sum()}")
print(f"Duplicate rows: {sales_data.duplicated().sum()}")