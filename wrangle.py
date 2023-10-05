# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Import numpy for numerical operations
import numpy as np

# Import Pandas for data manipulation 
import pandas as pd

# Import Matplotlib for data visualization
import matplotlib.pyplot as plt

# Import seaborn for data visualization
import seaborn as sns



# +

def acquire_retail(csv_file_path):
    """
    Read a CSV file into a DataFrame for retail data acquisition.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the retail data from the CSV file.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path, low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {str(e)}")
        return None




# +
def summarize(df) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    info
    shape
    outliers
    description
    missing data stats
    
    Args:
    df (DataFrame): The DataFrame to be summarized.
    k (float): The threshold for identifying outliers.
    
    return: None (prints to console)
    '''
    # print info on the df
    print('Shape of Data: ')
    print(df.shape)
    print('======================\n======================')
    print('Info: ')
    print(df.info())
    print('======================\n======================')
    
    
    # Calculate missing values and percentages
    missing_values = df.isnull()
    missing_count = missing_values.sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    print('Missing Data Stats:')
    print('Missing Data Count by Column:')
    print(missing_count)
    print('Missing Data Percentage by Column:')
    print(missing_percentage)
    
   
# -

def prepare_retail_data(df):
    """
    Prepare retail data for analysis by performing the following steps:
    1. Drop columns with high percentages of missing values.
    2. Drop rows with missing 'ship_date' values.
    3. Impute missing values for selected columns with their mode.
    4. Fill 'order_priority' with 'Not Specified'.
    5. Convert 'profit' column to numeric (float).
    6. Reassign the 'order_date' column to a datetime type.
    7. Convert the 'shipping_cost' column to numeric (float) with error handling.
    8. Impute missing 'shipping_cost' values with the mean.
    9. Calculate the median of the 'profit' column and fill missing values with it.
    10. Convert 'ship_date' column to datetime with errors='coerce' and filter out invalid dates.
    11. Convert 'customer_age' column to int.
    12. Convert selected categorical columns to category type.
    13. Convert numeric columns to float.
    14. Impute missing 'sales' values using forward-fill.
    15. Impute missing 'unit_price' values using forward-fill.
    16. Impute missing 'product_base_margin' values using mean imputation.

    Args:
    df (DataFrame): The DataFrame containing retail data.

    Returns:
    pd.DataFrame: The DataFrame with the specified data preparation steps applied.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Drop columns with high percentages of missing values
    df_copy.drop(columns=['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28'], inplace=True)

    # Drop rows with missing 'ship_date' values
    df_copy.dropna(subset=['ship_date'], inplace=True)

    # Impute missing values for selected columns
    columns_to_impute = ['city', 'customer_name', 'customer_segment', 'ship_mode', 'order_priority', 'product_name']
    for column in columns_to_impute:
        df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)

    # Fill 'order_priority' with 'Not Specified'
    df_copy['order_priority'].fillna('Not Specified', inplace=True)

    # Convert 'profit' column to numeric (float)
    df_copy['profit'] = pd.to_numeric(df_copy['profit'], errors='coerce')

    # Reassign the order_date column to be a datetime type
    df_copy['order_date'] = pd.to_datetime(df_copy['order_date'])

    # Convert the 'shipping_cost' column to numeric (float) with error handling
    df_copy['shipping_cost'] = pd.to_numeric(df_copy['shipping_cost'], errors='coerce')

    # Impute missing 'shipping_cost' values with the mean
    df_copy['shipping_cost'].fillna(df_copy['shipping_cost'].mean(), inplace=True)

    # Calculate the median of the 'profit' column
    median_profit = df_copy['profit'].median()

    # Fill missing values in 'profit' with the median
    df_copy['profit'].fillna(median_profit, inplace=True)

    # Convert 'ship_date' column to datetime with errors='coerce'
    df_copy['ship_date'] = pd.to_datetime(df_copy['ship_date'], errors='coerce')

    # Filter out rows with invalid dates (NaT)
    df_copy = df_copy[~df_copy['ship_date'].isna()]

    # Convert 'customer_age' column to int
    df_copy['customer_age'] = df_copy['customer_age'].astype(int)

    # Convert selected categorical columns to category type
    categorical_columns = ['order_priority', 'product_category', 'product_container', 'product_name', 'product_sub_category', 'region', 'state', 'zip_code', 'ship_mode']
    df_copy[categorical_columns] = df_copy[categorical_columns].astype('category')

    # Convert numeric columns to float
    numeric_columns = ['discount', 'order_quantity', 'product_base_margin', 'sales', 'shipping_cost', 'unit_price']
    df_copy[numeric_columns] = df_copy[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Impute missing 'sales' values using forward-fill
    df_copy['sales'].fillna(method='ffill', inplace=True)

    # Impute missing 'unit_price' values using forward-fill
    df_copy['unit_price'].fillna(method='ffill', inplace=True)

    # Impute missing 'product_base_margin' values using mean imputation
    mean_margin = df_copy['product_base_margin'].mean()
    df_copy['product_base_margin'].fillna(mean_margin, inplace=True)

    return df_copy



def sort_and_reset_index(df, column_name):
    """
    Sort a DataFrame by a specified column and reset the index.

    Args:
    df (pd.DataFrame): The DataFrame to be sorted.
    column_name (str): The name of the column to sort by.

    Returns:
    pd.DataFrame: The sorted DataFrame with the index reset.
    """
    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=column_name)

    # Reset the index to reflect the new order
    df_sorted.reset_index(drop=True, inplace=True)

    return df_sorted



def train_test_split_proportion(df, train_proportion=0.8):
    """
    Split a DataFrame into training and test sets based on the specified proportion.

    Args:
    df (DataFrame): The DataFrame to be split.
    train_proportion (float): The proportion of data to include in the training set (default is 0.8).

    Returns:
    tuple: A tuple containing the training DataFrame and test DataFrame.
    """
    # Calculate the size of the training set based on the proportion
    train_size = int(train_proportion * len(df))

    # Split the data into training and test sets
    train_data = df[:train_size]
    test_data = df[train_size:]

    return train_data, test_data



def display_numeric_column_histograms(data_frame):
    """
    Display histograms for numeric columns in a DataFrame with three colors.

    Args:
    data_frame (DataFrame): The DataFrame to visualize.

    Returns:
    None(prints to console)
    """
    numeric_columns = data_frame.select_dtypes(exclude=["object", "category"]).columns.to_list()
    # Define any number of colors for the histogram bars
    colors = ["blue"]
    for i, column in enumerate(numeric_columns):
        # Create a histogram for each numeric column with two colors
        figure, axis = plt.subplots(figsize=(10, 3))
        sns.histplot(data_frame, x=column, ax=axis, color=colors[i % len(colors)])
        axis.set_title(f"Histogram of {column}")
        plt.show()


# +


def plot_time_series(data_frame, x_column, y_column, title):
    """
    Create a time series plot from a DataFrame.

    Args:
    data_frame (DataFrame): The DataFrame containing the time series data.
    x_column (str): The name of the column to use for the x-axis (time).
    y_column (str): The name of the column to use for the y-axis (values).
    title (str): The title of the plot.

    Returns:
    None (displays the plot).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data_frame[x_column], data_frame[y_column])
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()



# -

def plot_time_series_multi(df, x_column, y_columns, title):
    """
    Create a time series plot from a DataFrame with blue lines.

    Args:
    df (DataFrame): The DataFrame containing the time series data.
    x_column (str): The name of the column to use for the x-axis (time).
    y_columns (list of str): A list of column names to plot on the y-axis.
    title (str): The title of the plot.

    Returns:
    None (displays the plot).
    """
    plt.figure(figsize=(12, 6))
    
    # Set the color to blue
    line_color = 'blue'
    
    # Plot multiple time series on the same plot with blue lines
    for column in y_columns:
        plt.plot(df[x_column], df[column], label=column, color=line_color)
    
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel('Values')
    
    # Add a legend to distinguish the lines
    plt.legend(loc='upper left')
    
    plt.show()

